# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE ADOPTION OF A PRESERVED CHECKPOINT BY A NEW CONFIGURATION (training_preflight.py, the module text; proxy
decision proxy-adopt-b1e-one-run-20261013 of Oct 13, 2026, section ADMISSION
CORRECTION REQUIRED: "Exercise the changed route on scratch records, including wrong predecessor, changed code/settings,
reused authorization, missing preservation and out-of-bound launch"; and the nine findings of the independent review of
commit 67c895be0, whose probes D:/UEMA_Research/external/D/UEMA_scratch/preflight_adoption/review/.../test_review_probes.py are carried here: each
GAP probe as a refusal test, each HOLD probe as a test).

The world of these tests mirrors the decision on stand-ins, in pytest's temporary folder: a preservation folder (its
MANIFEST.json recording the sha256 of every preserved file: the start checkpoint's meta.json, model.safetensors and
optimizer.pt, the latent body state latent_body/latest.pt and an episode companion; a copy of the run's stats and
stage_done.json); the predecessor run's own folder (stage_done.json ok at model step 1,000); its outcome receipt; the
decision's message (a JSON file whose "text" is hashed); the delegation record holding the operator's sentences; the
command template; a test registry with the adoption's two rows; and a new configuration's record, written with the
template's command resuming the preserved checkpoint, its seven gates judged FAIL. The adoption: one bounded run from
1,000 to 1,500. Nothing here reads or writes a production record, the production registry (the fixture
`no_production_registry` points the tool's default registry at a missing file of the test's own folder), Saved/ or
D:/UEMA_Research/external/D/UEMA_preserved.

THE ROUTE (ADOPTION_ROUTES): the fixture `pin` points the route at a test world's own pins for one test (pytest's
monkeypatch restores it); the test of (4) reads the route as written in the code, whose fingerprint the coordinator fills
after the configuration's record exists.

What is held (each refusal is checked when the adoption is written, at launch, and when an attempt or a launch is noted,
wherever that moment can see the condition: a launch command's own flags exist only at launch; the preserved files are
hashed when written and at launch):
  (1) without an adoption the configuration the route lists is never trained normally;
  (2) the adoption admits exactly its launch (launch_plan: kind continuation, block adoption, 1,000 -> 1,500); the
      record stays byte-identical, its gates FAIL; the attempt and the launch are recorded in the training row;
  (3) used once, held twice: the adoption file and the training row's record (GAP9: a hand-edited adoption file is
      refused by the row's record); a failed attempt (noted, never trained or marked) refuses a second launch;
  (4) THE ROUTE AND ITS PINS: the route as written; an entry not yet filled; a configuration the route does not name;
      each pinned value compared exactly (GAP2, GAP3, GAP5); HOLD_B (a same-step checkpoint of another run);
  (5) THE DECISION AND THE DELEGATION, read from their files (GAP3);
  (6) WRONG PREDECESSOR;
  (7) MISSING PRESERVATION, and every preserved file's sha256 (GAP6);
  (8) CHANGED CODE OR SETTINGS: the command template (GAP7), the launch's folders (GAP8), the configuration comparison
      (HOLD_A, HOLD_C, GAP10), the checkout's code, the record;
  (9) REUSED AUTHORIZATION, by containment as by equality (GAP4);
  (10) OUT-OF-BOUND LAUNCH;
  (11) THE GATES: every gate FAIL with its complete exception {failure, relevance, why_measurable, evidence} (GAP1,
      GAP1b), and the block's form;
  (12) THE REGISTRY ROWS (notes proxy-prediction-closure-20261013): only the pinned training row launches, a
      completion row never stands in for it and is only read; the training row's state and its record of a noted
      attempt (a failed one included: noted, never marked launched) refuse a second launch;
  (13) the noting needs src; the command line.
"""
import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest

from mimo_bridge.tools import training_preflight as TP

FLAGS = ("--intention", "--resume", "--out", "--report-every", "--max-steps", "--checkpoint-storage",
         "--proprio-token-skip", "--latent-body-output", "--latent-body-resume", "--latent-body-hierarchy",
         "--caregiver-record-dir", "--checkpoint-every-min", "--minutes")
TRAINER = ("import argparse\n\n\ndef parse_args(argv=None):\n    ap = argparse.ArgumentParser()\n"
           + "".join('    ap.add_argument("%s")\n' % f for f in FLAGS) + "    return ap.parse_args(argv)\n")
STOP_CRITERIA = [dict(name="vision_causal_error", path="causal_pe_per_modality.vision", op=">", line=0.04,
                      from_run_hour=0, mode="stop", note="S1")]
ROWS = dict(adopt_stop_row="written", adopt_done_row="queued", done_row="done", running_row="running",
            other_row="written")
DECISION = "proxy-adopt-test-20261013"
AUTHOR = "the overseer"
SENTENCES = TP.ADOPTION_ROUTES[0]["delegation_words"]         # the operator's quoted sentences, as the route pins them
AUTHORIZATION = ("\n".join('"%s"' % s for s in SENTENCES) + "\nDelegation reference: the operator's delegation of "
                 "Oct 13, 2026; proxy decision %s, by %s." % (DECISION, AUTHOR))
MESSAGE_TEXT = "H84 PROXY DECISION: %s\n(a stand-in for the decision's text)\n" % DECISION
FROM_COMMIT = "abcdef0123456789abcdef0123456789abcdef01"
WALL_CAP = 2000
OTHER_FP = "0123456789abcdef"
PRODUCTION_ROUTE = dict(
    checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/run/checkpoints/ckpt_step_489350",
    manifest="D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/MANIFEST.json",
    manifest_sha256="81b2cff358ecd17edd36d65648d71a26bb744c6aed5f8b1e3c26f470186e9572",
    preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
    model_sha256="69e9997d6c32cf22ad3bed76f47e1168fdce6e7d51f376071dd6162f72e7a397",
    latent_sha256="68ee27286b1f412b8c43e30ec4da598f12eb5a8683c0e7ffb26742e0ae742978",
    predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/run",
    outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/body1_extension/reading/outcome_receipt.json",
    outcome_receipt_sha256="da924702b729be24d83565f46d8e6aec072c722213df04ea5f140ed93980e11a",
    decision="proxy-adopt-b1e-one-run-20261013",
    decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                     "proxy-adopt-b1e-one-run-20261013/message.json",
    decision_text_sha256="5338b94e4ace134d049377b9514a59c2ddca86331af3b00926675b1bc18f7807",
    decision_author="the overseer",
    delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                      "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
    delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                      "You will determin launch conditions in my absense!"),
    from_commit="78717f9cd71b286099054f6de12749b60888c23e",
    command_template="D:/UEMA_Research/external/D/UEMA_scratch/body1_extension/command.json",
    command_template_sha256="21dcd5773966b7ca96693cd4edc2403513fc4e11b79b25133953318fdd035783",
    run_row="b1_adopt_B1A_skip13_489350",
    start=489350, end=494025, wall_cap_s=20683)
ARM_STAGE_ROUTE = dict(   # THE SECOND ENTRY (the module text): the arm-stage learning run
    checkpoint="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025",
    manifest="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/MANIFEST.json",
    manifest_sha256="b73e1e75c50214d1d478a0d67575fd7337560be224f9d5f40b02f9ea11555a11",
    preserved_root="D:/UEMA_Research/external/D/UEMA_preserved",
    model_sha256="8f99a3ce379303eb01c8364c9f40007a13e811c1b77f59b1b89c4058f44a315a",
    latent_sha256="6108b83ec565eb74ef493a770d377347cc83dfaeb10249fa1c07191ee7d660b4",
    predecessor_run="D:/UEMA_Research/weights/b1_continue_B1K_skip13_492800/run",
    outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_continue_492800/reading/outcome_receipt.json",
    outcome_receipt_sha256="1344f4f6e40665c3dce9d1f6e878e826d58dd4c0d3d6d93c130708c0241fab58",
    decision="proxy-arm-learning-package-corrections-20261014",
    decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                     "proxy-arm-learning-package-corrections-20261014/message.json",
    decision_text_sha256="a9739e9ef2ad50861bbe787a61c81d0d221e511df7bee8cc5b627c541b9f3a7c",
    decision_author="the overseer",
    delegation_record=PRODUCTION_ROUTE["delegation_record"],
    delegation_words=PRODUCTION_ROUTE["delegation_words"],
    from_commit="641c931a67edf2988ec886ed15c93c376cf18f46",
    command_template="D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_learning_494025/command.json",
    command_template_sha256="a6b71abf31b8a4599d1c6536047aef43804b3aafa204ad929f4b22b93421822e",
    run_row="arm_stage_learning_494025_run",
    start=494025, end=496525, wall_cap_s=5821)
ARM_STAGE_FINGERPRINT = "c02c8947283b0b02"
ARM_STAGE_R2_ROUTE = dict(ARM_STAGE_ROUTE,   # THE THIRD ENTRY (the module text): the linked replacement, the same start
                          decision="arm-reporting-device-fault-repair-20261015",
                          decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                                           "arm-reporting-device-fault-repair-20261015/message.json",
                          decision_text_sha256="6ded0daf9b88cc477fe30e959cce179e9b2dbc63b59684a063159c4bd52b6443",
                          command_template="D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_learning_494025_r2/command.json",
                          command_template_sha256="c2e8126dc77fc72072d489ffc4ae72e79cc483467977f5476904e481a0e4d286",
                          run_row="arm_stage_learning_494025_run2")
ARM_STAGE_R2_FINGERPRINT = "50372bae86b0724a"
ARM_STAGE_OFF_ROUTE = dict(ARM_STAGE_R2_ROUTE,   # THE FOURTH ENTRY (the module text): the stage-off arm, the same start
                           decision="arm-onoff-reader-first-preparation-decision-20261015",
                           decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                                            "arm-onoff-reader-first-preparation-decision-20261015/message.json",
                           decision_text_sha256="1b7b30ccb6468f96280e27897d312233aaf1d022fd74c9078c903f71f0d74199",
                           command_template="D:/UEMA_Research/external/D/UEMA_scratch/arm_stage_onoff_comparison_20261015/command.json",
                           command_template_sha256="8edab7ae54d1270eb72402670e9b9d2066afb52eb79e268e0b028b6fc76f2ed1",
                           run_row="arm_stage_onoff_off_run_494025",
                           wall_cap_s=7200)          # D4 amended (arm-off-prospective-cap7200-amendment-20261015)
ARM_STAGE_OFF_FINGERPRINT = "d1309fe930034fe4"
CK_REL = "run/checkpoints/ckpt_step_1000/"
PRESERVED = {CK_REL + "meta.json": json.dumps(dict(step=1000)).encode("utf-8"),
             CK_REL + "model.safetensors": b"the predecessor's weights at model step 1000",
             CK_REL + "optimizer.pt": b"the predecessor's optimizer state",
             "latent_body/latest.pt": b"the predecessor's latent body state",
             "episode_companion/ckpt_step_1000_0123.pt.gz": b"the predecessor's eight worlds"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _exceptions(gates=range(1, 8)):
    return {str(n): dict(failure="gate %d reads FAIL on the preserved checkpoint" % n,
                         relevance="the failure is shared by the run's start and its end",
                         why_measurable="the run's readings compare physical outcomes at its start and its end",
                         evidence=["row_%d" % n]) for n in gates}


def _manifest(pres: Path, files=None, with_stage_done=True, **more) -> bytes:
    """The preservation's MANIFEST.json over `files` ({relative path: bytes}, PRESERVED when None) and, with
    `with_stage_done`, the preserved run's stage_done.json as it reads on disk, as the preservation tool writes it
    ({files: [{path, bytes, sha256}], model_sha256, step, ...})."""
    files = dict(PRESERVED if files is None else files)
    done = pres / "run" / "stage_done.json"
    if with_stage_done and done.is_file():
        files["run/stage_done.json"] = done.read_bytes()
    data = dict(step=1000, model_sha256=_sha(PRESERVED[CK_REL + "model.safetensors"]), verdict="pass",
                files=[dict(path=rel, bytes=len(b), sha256=_sha(b)) for rel, b in files.items()])
    data.update(more)
    body = json.dumps(data, indent=1).encode("utf-8")
    (pres / "MANIFEST.json").write_bytes(body)
    return body


def _template(tmp_path) -> Path:
    """The command template: the configuration's flags, run flags the launch keeps, and the five locations and the bound
    that a launch replaces."""
    cmd = ["python", "-u", "-m", "mimo_bridge.train_mimo", "--intention", "on", "--report-every", "100",
           "--proprio-token-skip", "1,3", "--latent-body-hierarchy", "shadow", "--caregiver-record-dir",
           "E:/caregiver_store", "--checkpoint-every-min", "60.0", "--minutes", "0",
           "--resume", "E:/old/run/checkpoints/ckpt_step_500", "--out", "E:/old/run",
           "--checkpoint-storage", "E:/old/run/checkpoints", "--latent-body-resume", "E:/old/latent/latest.pt",
           "--latent-body-output", "E:/old/latent/output", "--max-steps", "1000"]
    p = tmp_path / "command_template.json"
    p.write_text(json.dumps(cmd, indent=1), encoding="utf-8")
    return p


def _world(tmp_path):
    """The stand-in preservation, predecessor run, receipt, decision message, delegation record, command template,
    registry and the new configuration's record (the module text)."""
    src, root, reg = _src(tmp_path), tmp_path / "pre", _registry(tmp_path)
    pres = tmp_path / "preserved"
    for rel, data in PRESERVED.items():
        (pres / rel).parent.mkdir(parents=True, exist_ok=True)
        (pres / rel).write_bytes(data)
    run = pres / "run"
    ck = run / "checkpoints" / "ckpt_step_1000"
    (run / "args.json").write_text(json.dumps(dict(max_steps=1000, resume=None)), encoding="utf-8")
    (run / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=1000, finished_at=time.time() - 100)),
                                         encoding="utf-8")
    _stats(run, ((900, 167.0), (1000, 440.0)))
    manifest = _manifest(pres)                               # the preserved run's stage_done.json among its files
    receipt = tmp_path / "outcome_receipt.json"
    receipt.write_text(json.dumps(dict(outcome="A", end=1000)), encoding="utf-8")
    message = tmp_path / "message.json"
    message.write_text(json.dumps(dict(id=DECISION, text=MESSAGE_TEXT, sha256=_sha(MESSAGE_TEXT.encode("utf-8")))),
                       encoding="utf-8")
    delegation = tmp_path / "delegation.txt"
    delegation.write_text("H84 OPERATOR DELEGATION (a stand-in)\n\n" + "\n\n".join('"%s"' % s for s in SENTENCES)
                          + "\n\nTHE APPLICATION OF THAT DELEGATION\n", encoding="utf-8")
    template = _template(tmp_path)
    latent, lat_out, out = pres / "latent_body" / "latest.pt", tmp_path / "lat_out", tmp_path / "new_run"
    cmd = TP.apply_sets(json.loads(template.read_text(encoding="utf-8")), [
        ("--resume", str(ck)), ("--out", str(out)), ("--checkpoint-storage", str(out / "checkpoints")),
        ("--latent-body-resume", str(latent)), ("--latent-body-output", str(lat_out)), ("--max-steps", "1500")])
    fp = Path(TP.new_record(cmd, "the new configuration (stand-in for main's code with body 1 excluded)", "tester",
                            root=root, src=src)).stem
    for n in range(1, 8):
        TP.set_gate(fp, n, "FAIL", "fails on the preserved checkpoint", ["row_%d" % n], "tester", root=root)
    time.sleep(0.01)
    return dict(src=src, root=root, reg=reg, fp=fp, cmd=cmd, ck=ck, run=run, pres=pres, latent=latent,
                lat_out=lat_out, out=out, manifest=pres / "MANIFEST.json",
                manifest_sha=_sha(manifest), model_sha=_sha(PRESERVED[CK_REL + "model.safetensors"]),
                latent_sha=_sha(PRESERVED["latent_body/latest.pt"]), receipt=receipt,
                receipt_sha=_sha(receipt.read_bytes()), message=message,
                message_sha=_sha(MESSAGE_TEXT.encode("utf-8")), delegation=delegation, template=template,
                template_sha=_sha(template.read_bytes()), tmp=tmp_path)


def _route(w, **changes):
    route = dict(fingerprint=w["fp"], checkpoint=str(w["ck"]), manifest=str(w["manifest"]),
                 manifest_sha256=w["manifest_sha"], preserved_root=str(w["pres"]), model_sha256=w["model_sha"],
                 latent_sha256=w["latent_sha"], predecessor_run=str(w["run"]), outcome_receipt=str(w["receipt"]),
                 outcome_receipt_sha256=w["receipt_sha"], decision=DECISION, decision_message=str(w["message"]),
                 decision_text_sha256=w["message_sha"], decision_author=AUTHOR, delegation_record=str(w["delegation"]),
                 delegation_words=SENTENCES, from_commit=FROM_COMMIT, command_template=str(w["template"]),
                 command_template_sha256=w["template_sha"], run_row="adopt_stop_row", start=1000, end=1500,
                 wall_cap_s=WALL_CAP)
    route.update(changes)
    return route


def _block(w, **more):
    b = dict(purpose="one bounded run of the new configuration from the preserved checkpoint, 1,000 to 1,500",
             predecessor=dict(checkpoint=str(w["ck"]), manifest_path=str(w["manifest"]),
                              manifest_sha256=w["manifest_sha"], model_sha256=w["model_sha"],
                              run=str(w["run"]), outcome_receipt=str(w["receipt"]),
                              outcome_receipt_sha256=w["receipt_sha"]),
             code_transition=dict(from_commit=FROM_COMMIT[:9], to_commit="2222222bbbb", commits=["2222222bbbb"]),
             decision=dict(id=DECISION, message_sha256=w["message_sha"], author=AUTHOR),
             authorization=AUTHORIZATION, authorized_by=AUTHOR, authorized_at="Oct 13, 2026 5:10 AM PDT",
             start_checkpoint=str(w["ck"]), rollback_checkpoint=str(w["ck"]),
             start_step=1000, end_step=1500, max_model_steps=500, wall_cap_s=WALL_CAP,
             run_flags={"--latent-body-resume": str(w["latent"]), "--latent-body-output": str(w["lat_out"])},
             stop_lines="S1", stop_lines_row="adopt_stop_row", stop_criteria=STOP_CRITERIA,
             completion_jobs_row="adopt_done_row",
             reachability="from 1,000 the first full stats record comes at 1,100; the stop line is first judged up to "
                          "246 model steps in, before the end at 1,500",
             no_auto_continue="the owner starts the trainer with --max-steps 1500 and no supervisor",
             exceptions=_exceptions())
    b.update(more)
    return b


def _pred(w, **changes):
    return dict(_block(w)["predecessor"], **changes)


@pytest.fixture(autouse=True)
def no_production_registry(monkeypatch, tmp_path):
    """The tool's default registry (note_attempt and mark_launched read it when none is given) is a file that does not
    exist, so no test can write production's registry."""
    monkeypatch.setattr(TP, "REGISTRY", tmp_path / "no_registry_given.json")


@pytest.fixture
def pin(monkeypatch):
    """pin(world, **changes): ADOPTION_ROUTES names the world's own adoption, with `changes` applied; restored after
    the test."""
    def _pin(w, **changes):
        monkeypatch.setattr(TP, "ADOPTION_ROUTES", (_route(w, **changes),))
        return w
    return _pin


def _write(w, **more):
    return TP.set_adoption(w["fp"], _block(w, **more), "tester", root=w["root"], registry=w["reg"], src=w["src"])


def _write_without(w, key):
    b = _block(w)
    b.pop(key)
    return TP.set_adoption(w["fp"], b, "tester", root=w["root"], registry=w["reg"], src=w["src"])


def _refusal(w, cmd=None):
    return TP.launch_refusal(cmd or w["cmd"], root=w["root"], src=w["src"], registry=w["reg"])


def _plan(w, cmd=None):
    return TP.launch_plan(cmd or w["cmd"], root=w["root"], src=w["src"])


def _folder(w, name):
    run = w["tmp"] / "experiments" / name
    run.mkdir(parents=True, exist_ok=True)
    return run


def _note(w, run):
    return TP.note_attempt(w["fp"], str(run), "launcher", root=w["root"], block=TP.ADOPTION, src=w["src"],
                           registry=w["reg"])


def _mark(w, run):
    return TP.mark_launched(w["fp"], str(run), "launcher", root=w["root"], block=TP.ADOPTION, src=w["src"],
                            registry=w["reg"])


def _held(w):
    return json.loads(TP.adoption_path(w["fp"], w["root"]).read_text(encoding="utf-8"))


def _edit_adoption(w, **changes):
    held = _held(w)
    held["block"].update(changes)
    TP._save(TP.adoption_path(w["fp"], w["root"]), held)


def _record(w):
    return TP._load(TP.record_path(w["fp"], w["root"]))


def _save_record(w, rec):
    TP._save(TP.record_path(w["fp"], w["root"]), rec)


def _rows(w):
    return {r["id"]: r for r in json.loads(w["reg"].read_text(encoding="utf-8"))["rows"]}


def _edit_row(w, row_id, **changes):
    data = json.loads(w["reg"].read_text(encoding="utf-8"))
    for r in data["rows"]:
        if r["id"] == row_id:
            r.update(changes)
    w["reg"].write_text(json.dumps(data), encoding="utf-8")


def _refused_written(w, needle, **more):
    """Writing the adoption (with `more` in its block) is refused with `needle`, and no adoption file is written."""
    with pytest.raises(TP.PreflightError) as exc:
        _write(w, **more)
    assert needle in str(exc.value), str(exc.value)
    assert not TP.adoption_path(w["fp"], w["root"]).exists()


def _refused_noted(w, needle):
    """Noting an attempt and marking a launch of the adoption are each refused with `needle`; nothing is recorded."""
    run = _folder(w, "run_noted")
    for fn in (_note, _mark):
        with pytest.raises(TP.PreflightError) as exc:
            fn(w, run)
        assert needle in str(exc.value), (fn.__name__, str(exc.value))
    assert not _held(w)["block"].get("attempts") and not _held(w)["block"].get("launched_at")
    assert not any(TP.ADOPTION_ROW_FIELD in r for r in _rows(w).values())


def _refused_at_launch_and_noted(w, needle, cmd=None):
    why = _refusal(w, cmd)
    assert why and needle in why, why
    _refused_noted(w, needle)


def _refused_everywhere(tmp_path, pin, needle, damage, **route_changes):
    """`damage` (applied to a fresh world before the adoption is written) refuses writing it; on a second world, applied
    after a clean write, it refuses the launch and the noting."""
    w = pin(_world(tmp_path / "written"))
    damage(w)
    pin(w, **route_changes)
    _refused_written(w, needle)
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    damage(w2)
    pin(w2, **route_changes)
    _refused_at_launch_and_noted(w2, needle)


LISTED = "is the adoption's (ADOPTION_ROUTES): it launches only as its adoption"
ATTEMPT = "the training row adopt_stop_row records this adoption's attempt"


# ------------------------------------------------------------------------------ (1) and (2)
def test_without_an_adoption_the_listed_configuration_is_never_trained_normally(tmp_path, pin):
    w = pin(_world(tmp_path))
    why = _refusal(w)
    assert why and LISTED in why, why
    assert _plan(w)["kind"] == "normal"
    pin(w, fingerprint=None)                                      # the route as written today: judged as before
    why = _refusal(w)
    assert why and "7 of 7 gates block launch" in why, why


def test_the_adoption_admits_exactly_its_launch_and_records_it_in_its_rows(tmp_path, pin):
    w = pin(_world(tmp_path))
    rec_path = TP.record_path(w["fp"], w["root"])
    before = rec_path.read_bytes()
    b = _write(w)
    assert rec_path.read_bytes() == before
    held = _held(w)
    assert held["fingerprint"] == w["fp"] and held["record_sha256"] == TP._sha(before)
    assert held["route"]["checkpoint"] == str(w["ck"]) and b["role"] == "adoption" and b["written_at_s"] > 0
    assert held["block"]["authorization"] == AUTHORIZATION and held["block"]["decision"]["id"] == DECISION
    assert _refusal(w) is None
    plan = _plan(w)
    assert (plan["kind"], plan["block"], plan["fingerprint"], plan["start_step"], plan["max_steps"]) == (
        "continuation", "adoption", w["fp"], 1000, 1500)
    assert plan["stop_criteria"] == STOP_CRITERIA
    assert plan["wall_cap_s"] == WALL_CAP          # the plan carries the block's wall cap (THE SECOND ENTRY)
    run = _folder(w, "run_a")
    _note(w, run)
    _edit_row(w, "adopt_stop_row", status="running")             # the run owner's job_started, between note and mark
    _mark(w, run)
    assert rec_path.read_bytes() == before                                  # noted and marked in the adoption file
    rows = _rows(w)
    held_row = rows["adopt_stop_row"][TP.ADOPTION_ROW_FIELD]                # the training row's record
    assert held_row["fingerprint"] == w["fp"] and held_row["decision"] == DECISION
    assert [a["run"] for a in held_row["attempts"]] == [str(run.resolve())] and held_row["launched_at"]
    assert rows["adopt_done_row"] == dict(id="adopt_done_row", status="queued")     # the completion row: only read
    rec = _record(w)
    assert [g["status"] for g in rec["gates"]] == ["FAIL"] * 7              # no gate qualified
    assert [s["checkpoint"] for s in rec["states"]] == [str(w["ck"])] and "repair_run" not in rec
    assert _held(w)["block"]["run"] == str(run.resolve())


# ------------------------------------------------------------------------------ (3) used once, held twice
def test_the_adoption_is_used_once_and_a_second_launch_after_the_first_was_marked_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    run = _folder(w, "run_x")
    _note(w, run)
    why = _refusal(w)
    assert why and ATTEMPT in why, why                                  # noted: no second launch, trained or not
    _stats(run, ((1100, 108.0),))
    why = _refusal(w)
    assert why and LISTED in why, why                                   # used: the same command is normal training
    assert _plan(w)["kind"] == "normal"
    with pytest.raises(TP.PreflightError, match="was used"):
        _write(w)
    _mark(w, run)
    with pytest.raises(TP.PreflightError, match="launched already"):
        _mark(w, run)
    with pytest.raises(TP.PreflightError, match="launched already"):
        _note(w, _folder(w, "run_y"))
    # a launch marked with no attempt noted uses it too: the second launch is refused
    w2 = pin(_world(tmp_path / "marked"))
    _write(w2)
    _mark(w2, _folder(w2, "run_m"))
    why = _refusal(w2)
    assert why and LISTED in why, why
    with pytest.raises(TP.PreflightError, match="launched already"):
        _note(w2, _folder(w2, "run_n"))


def test_GAP9_a_hand_edited_adoption_file_is_refused_by_its_registry_rows(tmp_path, pin):
    """The used-once memory is held twice: clearing the adoption file's launch and attempts leaves the rows' record."""
    w = pin(_world(tmp_path))
    _write(w)
    run = _folder(w, "run_first")
    _note(w, run)
    _mark(w, run)
    _stats(run, ((1100, 108.0),))
    new_out = str(w["lat_out"]) + "_second"
    _edit_adoption(w, launched_at=None, launched_at_s=None, run=None, attempts=[],
                   run_flags={"--latent-body-resume": str(w["latent"]), "--latent-body-output": new_out})
    why = _refusal(w, TP.apply_sets(w["cmd"], [("--latent-body-output", new_out)]))
    assert why and "the training row adopt_stop_row records this adoption's launch marked" in why, why
    with pytest.raises(TP.PreflightError, match="records this adoption's launch marked"):
        _note(w, _folder(w, "run_second"))
    with pytest.raises(TP.PreflightError, match="records this adoption's launch marked"):
        _mark(w, _folder(w, "run_second"))


def test_a_row_recording_another_adoption_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    _edit_row(w, "adopt_stop_row", **{TP.ADOPTION_ROW_FIELD: dict(fingerprint=OTHER_FP, attempts=[])})
    _refused_written(w, "the training row adopt_stop_row records the launch of another adoption")


def test_a_failed_attempt_refuses_a_second_launch(tmp_path, pin):
    """The notes proxy-prediction-closure-20261013: an attempt noted whose trainer never trained and was never
    marked launched (a failed attempt) still refuses a second launch, a rewrite of the adoption and another noting; the
    adoption file alone would count it as no use."""
    w = pin(_world(tmp_path))
    _write(w)
    run = _folder(w, "run_failed")
    _note(w, run)
    assert TP.adoption_available(_held(w)["block"])                     # the adoption file: no use
    new_out = w["tmp"] / "lat_out_retry"
    retry = TP.apply_sets(w["cmd"], [("--latent-body-output", str(new_out))])
    why = _refusal(w, retry)
    assert why and ATTEMPT in why, why
    with pytest.raises(TP.PreflightError, match=ATTEMPT):
        _write(w, run_flags={"--latent-body-resume": str(w["latent"]), "--latent-body-output": str(new_out)})
    with pytest.raises(TP.PreflightError, match=ATTEMPT):
        _note(w, _folder(w, "run_retry"))
    # the training row's state alone refuses too: the run owner's start and end of the failed attempt
    w2 = pin(_world(tmp_path / "state"))
    _write(w2)
    _note(w2, _folder(w2, "run_failed"))
    _edit_row(w2, "adopt_stop_row", status="failed", **{TP.ADOPTION_ROW_FIELD: None})   # its record removed by hand
    why = _refusal(w2)
    assert why and "the training row adopt_stop_row reads status failed" in why, why


def test_only_the_pinned_training_row_launches_and_a_completion_row_never_stands_in(tmp_path, pin):
    """The notes proxy-prediction-closure-20261013: only the exact authorized training row can launch; a completion
    row cannot be substituted as that row."""
    for more, needle in ((dict(stop_lines_row="adopt_done_row"),
                          "pins the training row adopt_stop_row; the adoption's stop_lines_row reads 'adopt_done_row'"),
                         (dict(stop_lines_row="other_row"),
                          "pins the training row adopt_stop_row; the adoption's stop_lines_row reads 'other_row'"),
                         (dict(completion_jobs_row="adopt_stop_row"),
                          "the adoption's completion_jobs_row names the training row adopt_stop_row")):
        name = more.get("stop_lines_row") or "completion"
        w = pin(_world(tmp_path / ("w_" + name)))
        _refused_written(w, needle, **more)
        _write(w)
        _edit_adoption(w, **more)
        _refused_at_launch_and_noted(w, needle)


def test_completion_rows_are_only_read(tmp_path, pin):
    """The completion jobs' row keeps the status its own jobs write: noting and marking write no field of it."""
    w = pin(_world(tmp_path))
    _write(w)
    before = _rows(w)["adopt_done_row"]
    run = _folder(w, "run_a")
    _note(w, run)
    _mark(w, run)
    assert _rows(w)["adopt_done_row"] == before


# ------------------------------------------------------------------------------ (4) THE ROUTE AND ITS PINS
def test_the_route_names_the_decisions_adoptions():
    """The route as written: seven entries, the first unchanged, the second the arm-stage learning run (the module
    text, THE SECOND ENTRY), the third its linked replacement (THE THIRD ENTRY), the fourth the stage-off arm of the
    on/off comparison (THE FOURTH ENTRY), the fifth the S4R learning-measurement continuation (THE FIFTH ENTRY), bound
    by runtime decision overnight-measurement-learning-authorized-20261022 (commit 6840506f7), the sixth the
    pose-return learning continuation (THE SIXTH ENTRY), bound by runtime decision
    overnight-pose-learning-authorized-20261022, the seventh its fresh attempt a2 (THE SEVENTH ENTRY), bound; the
    first two share no decision, training row, checkpoint, command template or fingerprint; the third shares the
    second's preserved start and nothing else of those; the fourth shares the third's preserved start, end and author
    and nothing else of those, with its own wall cap; the fifth and the sixth share no checkpoint, training row or
    command template with any other; the sixth pins the watcher's 5 s look and 30 s checks, which none of the first
    five pins; the seventh shares the sixth's preserved start, end, wall cap, code transition start, author and pinned
    timing, names its own training row, command template, fingerprint and decision."""
    assert len(TP.ADOPTION_ROUTES) == 9
    for _e, _branch, _fp, _dec, _sha in (
            (TP.ADOPTION_ROUTES[7], "control", "a58629bf99a000f5", "a2-consistency-pilot-authorized-20261023", "c4b12b15e2cb3e419a4f61dd5709164ccc7a60158461e962992c21994308efc0"),
            (TP.ADOPTION_ROUTES[8], "treatment", "6caef5b863c6c1c0", "a2-consistency-treatment-authorized-20261023", "37c49fb0d7ac42c04f38727bc47e4f571a2623cf5ad79f4453f112a6c683c8de")):
        # bound by runtime decisions, one per entry
        assert _e["fingerprint"] == _fp
        assert _e["decision"] == _dec
        assert _e["decision_text_sha256"] == _sha
        assert _e["run_row"] == "arm_consistency_496350_%s_run" % _branch
        assert _e["command_template"].endswith("command_%s.json" % _branch)
    assert TP.adoption_route("a58629bf99a000f5") is TP.ADOPTION_ROUTES[7]
    assert TP.adoption_route("6caef5b863c6c1c0") is TP.ADOPTION_ROUTES[8]
    route = dict(TP.ADOPTION_ROUTES[0])
    fingerprint = route.pop("fingerprint")
    assert route == PRODUCTION_ROUTE
    # filled by the coordinator in a reviewed one-line change once the configuration's record exists
    assert fingerprint is None or TP._is_fingerprint(fingerprint)
    arm = dict(TP.ADOPTION_ROUTES[1])
    assert arm.pop("fingerprint") == ARM_STAGE_FINGERPRINT and arm == ARM_STAGE_ROUTE
    for key in ("decision", "run_row", "checkpoint", "command_template"):
        assert ARM_STAGE_ROUTE[key] != PRODUCTION_ROUTE[key], key
    assert TP.adoption_route(ARM_STAGE_FINGERPRINT) is TP.ADOPTION_ROUTES[1]
    r2 = dict(TP.ADOPTION_ROUTES[2])
    assert r2.pop("fingerprint") == ARM_STAGE_R2_FINGERPRINT and r2 == ARM_STAGE_R2_ROUTE
    for key in ("checkpoint", "manifest", "manifest_sha256", "model_sha256", "latent_sha256", "start", "end",
                "wall_cap_s"):
        assert ARM_STAGE_R2_ROUTE[key] == ARM_STAGE_ROUTE[key], key
    for key in ("decision", "decision_text_sha256", "run_row", "command_template", "command_template_sha256"):
        assert ARM_STAGE_R2_ROUTE[key] != ARM_STAGE_ROUTE[key], key
    assert ARM_STAGE_R2_FINGERPRINT != ARM_STAGE_FINGERPRINT
    assert TP.adoption_route(ARM_STAGE_R2_FINGERPRINT) is TP.ADOPTION_ROUTES[2]
    off = dict(TP.ADOPTION_ROUTES[3])
    assert off.pop("fingerprint") == ARM_STAGE_OFF_FINGERPRINT and off == ARM_STAGE_OFF_ROUTE
    for key in ("checkpoint", "manifest", "manifest_sha256", "model_sha256", "latent_sha256", "start", "end"):
        assert ARM_STAGE_OFF_ROUTE[key] == ARM_STAGE_R2_ROUTE[key], key
    for key in ("decision", "decision_text_sha256", "run_row", "command_template", "command_template_sha256",
                "wall_cap_s"):
        assert ARM_STAGE_OFF_ROUTE[key] != ARM_STAGE_R2_ROUTE[key], key
    assert ARM_STAGE_OFF_FINGERPRINT not in (ARM_STAGE_FINGERPRINT, ARM_STAGE_R2_FINGERPRINT)
    assert TP.adoption_route(ARM_STAGE_OFF_FINGERPRINT) is TP.ADOPTION_ROUTES[3]
    s4r = TP.ADOPTION_ROUTES[4]
    # bound by the runtime decision (commit 6840506f7): fingerprint, decision and its message's text sha256
    assert s4r["fingerprint"] == "b4ebcf3490cbd4fb"
    assert s4r["decision"] == "overnight-measurement-learning-authorized-20261022"
    assert s4r["decision_text_sha256"] == "0f76c0b4340f85a602d49d88d6d8064e8f77f97cc7843f500fd1328c5e4aa911"
    assert TP.adoption_route("b4ebcf3490cbd4fb") is s4r
    assert (s4r["start"], s4r["end"], s4r["wall_cap_s"]) == (496000, 496250, 1200)
    for other in TP.ADOPTION_ROUTES[:4]:
        for key in ("checkpoint", "run_row", "command_template", "command_template_sha256", "manifest_sha256"):
            assert s4r[key] != other[key], key
    pose = TP.ADOPTION_ROUTES[5]
    # bound by the runtime decision overnight-pose-learning-authorized-20261022: fingerprint, decision and its text sha256
    assert pose["fingerprint"] == "2819b74cc6afc95c"
    assert pose["decision"] == "overnight-pose-learning-authorized-20261022"
    assert pose["decision_text_sha256"] == "c9c3061ddeacf98bb59f849099cb8e2eaaf432b074fd2835375ed00e72178611"
    assert TP.adoption_route("2819b74cc6afc95c") is pose
    assert (pose["start"], pose["end"], pose["wall_cap_s"]) == (496250, 496350, 720)
    assert (pose["watch_poll_s"], pose["watch_first_check_s"]) == (5, 30)
    for other in TP.ADOPTION_ROUTES[:5]:
        assert "watch_poll_s" not in other and "watch_first_check_s" not in other
        for key in ("checkpoint", "run_row", "command_template", "command_template_sha256", "manifest_sha256"):
            assert pose[key] != other[key], key
    a2 = TP.ADOPTION_ROUTES[6]
    # the fresh attempt a2 (pose-learning-repairs-decisions-20261022 B2 and
    # pose-package-a2-preparation-decision-20261022 item 5), bound by runtime decision
    # a2-bounded-learning-authorized-20261023: fingerprint, decision and its text sha256
    assert a2["fingerprint"] == "83ff9dd7eb5e2699"
    assert a2["decision"] == "a2-bounded-learning-authorized-20261023"
    assert a2["decision_text_sha256"] == "c239d77e469c4daa2cc9443062dc7b1ba15f4b3f4a0ed0445867ec7a600225e4"
    assert TP.adoption_route("83ff9dd7eb5e2699") is a2
    assert (a2["start"], a2["end"], a2["wall_cap_s"]) == (496250, 496350, 720)
    assert (a2["watch_poll_s"], a2["watch_first_check_s"]) == (5, 30)
    for key in ("checkpoint", "manifest", "manifest_sha256", "model_sha256", "latent_sha256", "predecessor_run",
                "outcome_receipt", "outcome_receipt_sha256", "from_commit", "decision_author"):
        assert a2[key] == pose[key], key
    for other in TP.ADOPTION_ROUTES[:6]:
        for key in ("run_row", "command_template", "command_template_sha256"):
            assert a2[key] != other[key], key


# THE SIXTH ENTRY'S WATCHER TIMING (the module text; the overnight-pose-learning-monitor-timing-decision-20261022
# and overnight-pose-learning-adoption-check-authorized-20261022): an adoption's stop-line reachability is computed with
# its route's pinned look and check seconds (adoption_watch_timing), only when the route pins them and the block names
# the same (watch_timing). In this world the start run's fastest pace is 100 model steps in 273 s and the first full
# stats record comes 100 model steps after the start: the default 30 s look and 300 s check judge the stop line within
# 100 + ceil(100/273 x 330) + 25 = 246 model steps, the pinned 5 s and 30 s within 100 + ceil(100/273 x 35) + 25 = 138.
SHORT_TIMING = dict(watch_poll_s=5, watch_first_check_s=30)
SHORT_BLOCK = dict(end_step=1200, max_model_steps=200)
UNREACHED = "before the watcher first judges stop line vision_causal_error"


def test_the_stop_lines_first_reading_uses_the_pinned_timing_and_the_default_elsewhere(tmp_path):
    w = _world(tmp_path)
    block = _block(w, **SHORT_BLOCK)
    assert TP.first_readings(block) == {"vision_causal_error": 246}
    assert TP.first_readings(block, None, (5, 30)) == {"vision_causal_error": 138}
    assert any(UNREACHED in p for p in TP.first_reading_problems(block))
    assert TP.first_reading_problems(block, None, None, (5, 30)) == []


def test_the_default_timing_still_refuses_a_run_too_short_for_its_stop_line(tmp_path, pin):
    w = pin(_world(tmp_path), end=1200)
    _refused_written(w, UNREACHED, **SHORT_BLOCK)


def test_the_pinned_short_timing_admits_the_reachable_bound_with_the_same_stop_inputs(tmp_path, pin):
    w = pin(_world(tmp_path), end=1200, **SHORT_TIMING)
    _write(w, watch_timing=dict(poll_s=5, first_check_s=30), **SHORT_BLOCK)
    assert _held(w)["block"]["stop_criteria"] == STOP_CRITERIA
    assert TP.adoption_watch_timing(w["fp"]) == (5, 30)
    cmd = TP.apply_sets(w["cmd"], [("--max-steps", "1200")])
    assert _refusal(w, cmd) is None


@pytest.mark.parametrize("timing", [None, dict(poll_s=5, first_check_s=60), dict(poll_s=30, first_check_s=30)])
def test_a_missing_or_mismatched_timing_is_refused(tmp_path, pin, timing):
    w = pin(_world(tmp_path), end=1200, **SHORT_TIMING)
    more = dict(SHORT_BLOCK, **({} if timing is None else dict(watch_timing=timing)))
    _refused_written(w, "pins the watcher's first-record look 5 s and checks 30 s", **more)


def test_a_route_that_pins_no_timing_cannot_borrow_the_short_timing(tmp_path, pin):
    w = pin(_world(tmp_path), end=1200)
    assert TP.adoption_watch_timing(w["fp"]) == (TP.WATCH_POLL_S, TP.WATCH_FIRST_CHECK_S)
    _refused_written(w, "pins no watcher timing", watch_timing=dict(poll_s=5, first_check_s=30), **SHORT_BLOCK)
    _refused_written(w, UNREACHED, watch_timing=dict(poll_s=5, first_check_s=30), **SHORT_BLOCK)


def test_a_route_entry_not_yet_filled_admits_nothing_when_written_at_launch_and_when_noted(tmp_path, pin):
    w = pin(_world(tmp_path), fingerprint=None)
    _refused_written(w, "is not one (another adoption is a new decision")
    pin(w)
    _write(w)
    pin(w, fingerprint=None)
    _refused_at_launch_and_noted(w, "an entry whose fingerprint is not yet filled (None, admitting nothing)")


def test_a_configuration_the_route_does_not_name_is_refused_when_written_at_launch_and_when_noted(tmp_path, pin):
    w = pin(_world(tmp_path), fingerprint=OTHER_FP)
    _refused_written(w, "configuration %s is not one" % w["fp"])
    pin(w)
    _write(w)
    pin(w, fingerprint=OTHER_FP)
    _refused_at_launch_and_noted(w, "configuration %s is not one" % w["fp"])


PIN_CHANGES = [
    ("checkpoint", lambda w: str(w["run"] / "checkpoints" / "ckpt_step_900"), "admits only the preserved checkpoint"),
    ("manifest", lambda w: str(w["tmp"] / "elsewhere" / "MANIFEST.json"), "pins the preservation MANIFEST"),
    ("manifest_sha256", lambda w: "0" * 64, "admits only the preservation whose MANIFEST reads sha256 " + "0" * 64),
    ("model_sha256", lambda w: "1" * 64, "pins the model sha256 " + "1" * 64),
    ("latent_sha256", lambda w: "2" * 64, "pins the latent body state's sha256 " + "2" * 64),
    ("predecessor_run", lambda w: str(w["tmp"] / "elsewhere" / "run"), "pins the predecessor run"),
    ("outcome_receipt", lambda w: str(w["tmp"] / "elsewhere" / "receipt.json"), "pins the predecessor's outcome receipt"),
    ("outcome_receipt_sha256", lambda w: "3" * 64, "pins the outcome receipt sha256 " + "3" * 64),
    ("decision", lambda w: "another-decision", "pins the decision another-decision; the adoption names " + DECISION),
    ("decision_text_sha256", lambda w: "4" * 64, "pins the decision message's text sha256 " + "4" * 64),
    ("decision_author", lambda w: "someone else", "pins the decision's author 'someone else'"),
    ("wall_cap_s", lambda w: 20683, "pins the wall cap 20683 s; the adoption's wall_cap_s reads %d" % WALL_CAP),
    ("start", lambda w: 999, "starts at model step 999; the adoption's start_step reads 1000"),
    ("end", lambda w: 1400, "ends at model step 1400; the adoption's end_step reads 1500"),
    ("from_commit", lambda w: "9" * 40, "pins the code transition's start, commit " + "9" * 40),
    ("run_row", lambda w: "other_row", "pins the training row other_row; the adoption's stop_lines_row reads "
                                       "'adopt_stop_row'"),
]


@pytest.mark.parametrize("key, value, needle", PIN_CHANGES, ids=[c[0] for c in PIN_CHANGES])
def test_each_pinned_value_is_compared_exactly(tmp_path, pin, key, value, needle):
    """GAP2 (the wall cap), GAP3 (a made-up decision, message hash or author), GAP5 (a MANIFEST copy elsewhere, another
    predecessor run): the route pins each value the decision names; a block that differs from one pin is refused when
    written, at launch and when noted."""
    w = pin(_world(tmp_path / "written"))
    pin(w, **{key: value(w)})
    _refused_written(w, needle)
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    pin(w2, **{key: value(w2)})
    _refused_at_launch_and_noted(w2, needle)


def test_the_code_transition_names_the_pinned_commit_by_its_hash_or_a_prefix_of_nine_or_more(tmp_path, pin):
    w = pin(_world(tmp_path))
    for named in (FROM_COMMIT[:8], FROM_COMMIT.upper()[:12], "f" + FROM_COMMIT[1:12]):
        _refused_written(w, "pins the code transition's start", code_transition=dict(
            from_commit=named, to_commit="2222222bbbb", commits=["2222222bbbb"]))
    _write(w, code_transition=dict(from_commit=FROM_COMMIT, to_commit="2222222bbbb", commits=["2222222bbbb"]))


def test_GAP2_the_decisions_wall_cap_only(tmp_path, pin):
    _refused_written(pin(_world(tmp_path)), "pins the wall cap 2000 s; the adoption's wall_cap_s reads 1000000000.0",
                     wall_cap_s=1.0e9)


def test_GAP5_a_manifest_copy_outside_the_preservation_and_another_predecessor_run_are_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "run").mkdir(parents=True)
    shutil.copyfile(w["manifest"], elsewhere / "MANIFEST.json")
    shutil.copyfile(w["run"] / "stage_done.json", elsewhere / "run" / "stage_done.json")
    w["manifest"].unlink()
    with pytest.raises(TP.PreflightError) as exc:
        _write(w, predecessor=_pred(w, manifest_path=str(elsewhere / "MANIFEST.json"), run=str(elsewhere / "run")))
    assert "pins the preservation MANIFEST" in str(exc.value) and "pins the predecessor run" in str(exc.value)


def test_HOLD_B_a_same_step_checkpoint_of_another_run_is_refused_by_the_route(tmp_path, pin):
    """The non-preserved original: same model step, same bytes, another run folder (the reviewer's mutation MB: the
    route's checkpoint compared by model step only)."""
    w = pin(_world(tmp_path))
    original_run = tmp_path / "copy_of_the_original" / "run"
    shutil.copytree(w["run"], original_run)
    original = original_run / "checkpoints" / "ckpt_step_1000"
    TP.add_state(w["fp"], str(original), "the original, same bytes", "tester", root=w["root"])
    _refused_written(w, "admits only the preserved checkpoint", start_checkpoint=str(original),
                     rollback_checkpoint=str(original), predecessor=_pred(w, checkpoint=str(original)))


# ------------------------------------------------------------------------------ (5) THE DECISION AND THE DELEGATION
def test_GAP3_a_made_up_decision_id_message_hash_author_and_authorization_are_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    with pytest.raises(TP.PreflightError) as exc:
        _write(w, decision=dict(id="made-up-id", message_sha256="a" * 64, author="anyone"), authorization="made-up-id")
    text = str(exc.value)
    for needle in ("pins the decision %s; the adoption names made-up-id" % DECISION,
                   "pins the decision message's text sha256", "pins the decision's author",
                   "the adoption's authorization does not hold the operator's delegation sentence"):
        assert needle in text, (needle, text)


@pytest.mark.parametrize("damage, needle", [
    (lambda w: w["message"].unlink(), "the decision's message %s cannot be read"),
    (lambda w: w["message"].write_text(json.dumps(dict(id="another", text=MESSAGE_TEXT)), encoding="utf-8"),
     "does not name decision %s (it names another)" % DECISION),
    (lambda w: w["message"].write_text(json.dumps(dict(id=DECISION, text=MESSAGE_TEXT + " edited")), encoding="utf-8"),
     "holds a text whose sha256 reads"),
    (lambda w: w["delegation"].unlink(), "the delegation record %s cannot be read"),
    (lambda w: w["delegation"].write_text(w["delegation"].read_text(encoding="utf-8").replace(SENTENCES[2], "x"),
                                          encoding="utf-8"),
     "does not hold the operator's sentence %r the route pins" % SENTENCES[2]),
], ids=["message-missing", "message-id", "message-text", "delegation-missing", "delegation-sentence"])
def test_the_decision_message_and_the_delegation_record_are_read(tmp_path, pin, damage, needle):
    w = pin(_world(tmp_path / "written"))
    expected = needle % (w["message"] if "message %s" in needle else w["delegation"]) if "%s cannot" in needle \
        else needle
    damage(w)
    _refused_written(w, expected)
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    expected = needle % (w2["message"] if "message %s" in needle else w2["delegation"]) if "%s cannot" in needle \
        else needle
    damage(w2)
    _refused_at_launch_and_noted(w2, expected)


def test_the_authorization_holds_each_delegation_sentence_verbatim(tmp_path, pin):
    w = pin(_world(tmp_path))
    for sentence in SENTENCES:
        _refused_written(w, "does not hold the operator's delegation sentence %r verbatim" % sentence,
                         authorization=AUTHORIZATION.replace(sentence, sentence.replace(" ", "  ", 1)))
    _write(w)
    _edit_adoption(w, authorization=AUTHORIZATION.replace(SENTENCES[0], SENTENCES[0].lower()))
    _refused_at_launch_and_noted(w, "does not hold the operator's delegation sentence %r verbatim" % SENTENCES[0])


# ------------------------------------------------------------------------------ (6) WRONG PREDECESSOR
def test_the_launch_resumes_exactly_the_preserved_checkpoint(tmp_path, pin):
    """launch_refusal routes a launch to the adoption only when its --resume is the adoption's start, so the launch-time
    comparison of --resume is read here directly; through launch_refusal another --resume is normal training."""
    w = pin(_world(tmp_path))
    _write(w)
    other = w["run"] / "checkpoints" / "ckpt_step_900"
    other.mkdir()
    cmd = TP.apply_sets(w["cmd"], [("--resume", str(other))])
    why = _refusal(w, cmd)
    assert why and LISTED in why, why
    assert _plan(w, cmd)["kind"] == "normal"
    why = TP._adoption_refusal(cmd, _record(w), _held(w), TP.configuration(cmd, w["src"]), w["root"], w["reg"])
    assert why and "not the adoption's start" in why, why


def test_a_predecessor_checkpoint_that_is_not_the_start_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    other = w["run"] / "checkpoints" / "ckpt_step_900"
    _refused_written(w, "is not its predecessor's checkpoint", predecessor=_pred(w, checkpoint=str(other)))
    _write(w)
    _edit_adoption(w, predecessor=_pred(w, checkpoint=str(other)))
    _refused_at_launch_and_noted(w, "is not its predecessor's checkpoint")


def test_a_start_step_that_is_not_the_start_checkpoints_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path), start=1001)
    _refused_written(w, "names model step 1000; the adoption's start_step reads 1001", start_step=1001,
                     max_model_steps=499)
    pin(w)
    _write(w)
    pin(w, start=1001)
    _edit_adoption(w, start_step=1001, max_model_steps=499)
    _refused_at_launch_and_noted(w, "names model step 1000; the adoption's start_step reads 1001")


def test_a_predecessor_model_whose_sha256_differs_is_refused_when_written_at_launch_and_when_noted(tmp_path, pin):
    _refused_everywhere(tmp_path, pin, "the predecessor's model",
                        lambda w: (w["ck"] / "model.safetensors").write_bytes(b"other weights"))


def test_a_manifest_recording_another_model_is_refused(tmp_path, pin):
    w = _world(tmp_path)
    body = _manifest(w["pres"], model_sha256=_sha(b"another model"))
    pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "records model sha256 %s; the adoption names %s" % (_sha(b"another model"), w["model_sha"]),
                     predecessor=_pred(w, manifest_sha256=_sha(body)))
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    _manifest(w2["pres"], model_sha256=_sha(b"another model"))
    _refused_at_launch_and_noted(w2, "records model sha256 %s" % _sha(b"another model"))


def test_a_latent_body_state_the_manifest_does_not_record_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    other = tmp_path / "other_latest.pt"
    other.write_bytes(b"another latent body state")
    _refused_written(w, "for the predecessor's latent_body/latest.pt",
                     run_flags={"--latent-body-resume": str(other), "--latent-body-output": str(w["lat_out"])})
    _write(w)
    w["latent"].write_bytes(b"the preserved state rewritten")
    _refused_at_launch_and_noted(w, "for the predecessor's latent_body/latest.pt")
    # a MANIFEST that records no latent body state
    w2 = _world(tmp_path / "unrecorded")
    body = _manifest(w2["pres"], {k: v for k, v in PRESERVED.items() if not k.startswith("latent_body/")})
    pin(w2, manifest_sha256=_sha(body))
    _refused_written(w2, "records no latent_body/latest.pt", predecessor=_pred(w2, manifest_sha256=_sha(body)))


@pytest.mark.parametrize("done", [dict(ok=False, steps=1000), dict(ok=True, steps=999), None],
                         ids=["not-ok", "other-step", "missing"])
def test_a_predecessor_run_that_did_not_end_at_the_start_step_is_refused(tmp_path, pin, done):
    def damage(w):
        if done is None:
            (w["run"] / "stage_done.json").unlink()
        else:
            (w["run"] / "stage_done.json").write_text(json.dumps(dict(done, finished_at=1.0)), encoding="utf-8")
    _refused_everywhere(tmp_path, pin, "did not end with its own save at model step 1000", damage)


def test_the_predecessor_runs_stage_done_is_the_one_the_manifest_records(tmp_path, pin):
    """The coordinator's decision on the package review's finding 2: the predecessor run is the preservation's copy, and
    its stage_done.json (still ok at the start step) is checked against the MANIFEST entry when written, at launch and
    when noted."""
    def edited(w):
        done = json.loads((w["run"] / "stage_done.json").read_text(encoding="utf-8"))
        (w["run"] / "stage_done.json").write_text(json.dumps(dict(done, note="edited after the preservation")),
                                                  encoding="utf-8")
    _refused_everywhere(tmp_path / "edited", pin, "the predecessor run's stage_done.json", edited)
    w = _world(tmp_path / "unrecorded")
    body = _manifest(w["pres"], with_stage_done=False)
    pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "records no run/stage_done.json, the predecessor run's stage_done.json",
                     predecessor=_pred(w, manifest_sha256=_sha(body)))


def test_a_predecessor_run_outside_the_preservation_folder_is_refused(tmp_path, pin):
    w = _world(tmp_path)
    outside = tmp_path / "original_run_folder"
    outside.mkdir()
    shutil.copyfile(w["run"] / "stage_done.json", outside / "stage_done.json")
    pin(w, predecessor_run=str(outside))
    _refused_written(w, "the predecessor run %s is not inside the preservation folder %s" % (outside, w["pres"]),
                     predecessor=_pred(w, run=str(outside)))


def test_an_outcome_receipt_whose_sha256_differs_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    w["receipt"].write_text("{}", encoding="utf-8")
    _refused_written(w, "the predecessor's outcome receipt")
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    w2["receipt"].unlink()
    _refused_at_launch_and_noted(w2, "the predecessor's outcome receipt %s reads nothing (not a file)" % w2["receipt"])


# ------------------------------------------------------------------------------ (7) MISSING PRESERVATION
def test_a_missing_manifest_is_refused_when_written_at_launch_and_when_noted(tmp_path, pin):
    w = pin(_world(tmp_path))
    w["manifest"].unlink()
    _refused_written(w, "the preservation MANIFEST %s is missing" % w["manifest"])
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    w2["manifest"].unlink()
    _refused_at_launch_and_noted(w2, "the preservation MANIFEST %s is missing" % w2["manifest"])


@pytest.mark.parametrize("content, needle", [(b"{not json", "cannot be read as JSON"),
                                             (b"[1, 2]", "is not a JSON object")], ids=["json", "object"])
def test_an_unreadable_manifest_is_refused(tmp_path, pin, content, needle):
    _refused_everywhere(tmp_path, pin, needle, lambda w: w["manifest"].write_bytes(content))


def test_a_manifest_reading_another_sha256_is_refused(tmp_path, pin):
    _refused_everywhere(tmp_path, pin, "(the preservation record changed, or it is another)",
                        lambda w: w["manifest"].write_bytes(w["manifest"].read_bytes() + b"\n"))


@pytest.mark.parametrize("name", TP.CHECKPOINT_FILES)
def test_a_predecessor_checkpoint_without_its_files_is_refused(tmp_path, pin, name):
    _refused_everywhere(tmp_path, pin, "holds no %s: the preserved state is incomplete" % name,
                        lambda w: (w["ck"] / name).unlink())


@pytest.mark.parametrize("rel, how", [(CK_REL + "optimizer.pt", "deleted"), (CK_REL + "optimizer.pt", "replaced"),
                                      ("episode_companion/ckpt_step_1000_0123.pt.gz", "replaced"),
                                      (CK_REL + "meta.json", "replaced")],
                         ids=["optimizer-deleted", "optimizer-replaced", "companion-replaced", "meta-replaced"])
def test_GAP6_every_preserved_file_the_manifest_records_is_hashed_when_written_and_at_launch(tmp_path, pin, rel, how):
    def damage(w):
        if how == "deleted":
            (w["pres"] / rel).unlink()
        else:
            (w["pres"] / rel).write_bytes(b"other bytes")
    needle = "the preserved file %s"
    w = pin(_world(tmp_path / "written"))
    damage(w)
    _refused_written(w, needle % (w["pres"] / rel))
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    damage(w2)
    why = _refusal(w2)
    assert why and needle % (w2["pres"] / rel) in why, why


def test_a_start_checkpoint_outside_the_preservation_folder_is_refused(tmp_path, pin):
    w = _world(tmp_path)
    outside = tmp_path / "copy_outside" / "run"
    shutil.copytree(w["run"], outside)
    ck = outside / "checkpoints" / "ckpt_step_1000"
    pin(w, checkpoint=str(ck))
    TP.add_state(w["fp"], str(ck), "a copy outside the preservation folder", "tester", root=w["root"])
    _refused_written(w, "the predecessor checkpoint %s is not inside the preservation folder %s" % (ck, w["pres"]),
                     start_checkpoint=str(ck), rollback_checkpoint=str(ck), predecessor=_pred(w, checkpoint=str(ck)))


def test_a_manifest_recording_no_companion_is_refused(tmp_path, pin):
    w = _world(tmp_path)
    body = _manifest(w["pres"], {k: v for k, v in PRESERVED.items() if not k.startswith("episode_companion/")})
    pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "records no file of episode_companion/", predecessor=_pred(w, manifest_sha256=_sha(body)))


# ------------------------------------------------------------------------------ (8) CHANGED CODE OR SETTINGS
def test_a_command_whose_configuration_differs_is_refused_at_launch_naming_each_difference(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    flag = TP.apply_sets(w["cmd"], [("--proprio-token-skip", "3")])
    why = _refusal(w, flag)
    assert why and "is not the adoption's; an adoption runs exactly its record's code and settings" in why, why
    assert "flag --proprio-token-skip: 1,3 -> 3" in why
    assert _plan(w, flag)["kind"] == "normal"
    (w["src"] / "lem" / "model" / "core.py").write_text("x = 2\n", encoding="utf-8")
    why = _refusal(w)
    assert why and "is not the adoption's" in why and "code: 1 file(s) differ: lem/model/core.py" in why, why
    # once the adoption is used, the same commands are judged exactly as before (no record of their configuration)
    (w["src"] / "lem" / "model" / "core.py").write_text("x = 1\n", encoding="utf-8")
    _mark(w, _folder(w, "run_used"))
    why = _refusal(w, flag)
    assert why and "has no preflight record" in why and "is not the adoption's" not in why, why


def test_HOLD_A_another_launch_from_another_checkpoint_is_judged_as_before_while_the_adoption_is_unused(tmp_path, pin):
    """The reviewer's mutation MA: the configuration comparison without its --resume test."""
    w = pin(_world(tmp_path))
    _write(w)
    other = w["run"] / "checkpoints" / "ckpt_step_900"
    other.mkdir()
    why = _refusal(w, TP.apply_sets(w["cmd"], [("--proprio-token-skip", "3"), ("--resume", str(other))]))
    assert why and "has no preflight record" in why and "is not the adoption's" not in why, why


def test_HOLD_C_another_configuration_with_its_own_record_is_refused_from_the_adoptions_start(tmp_path, pin):
    """The reviewer's mutation MC: the configuration comparison skipped when the command's configuration has a record."""
    w = pin(_world(tmp_path))
    _write(w)
    other_cmd = TP.apply_sets(w["cmd"], [("--proprio-token-skip", "3")])
    fp_b = Path(TP.new_record(other_cmd, "another configuration", "tester", root=w["root"], src=w["src"])).stem
    for n in range(1, 8):
        TP.set_gate(fp_b, n, "PASS", "stand-in", ["e"], "tester", root=w["root"])
    why = _refusal(w, other_cmd)
    assert why and "is not the adoption's" in why, why


def test_GAP10_an_unreadable_adoption_file_concerns_only_launches_from_its_checkpoint(tmp_path, pin):
    """--resume is compared with the route's checkpoint before the route's adoption file is read."""
    w = pin(_world(tmp_path))
    _write(w)
    TP.adoption_path(w["fp"], w["root"]).write_text("{not json", encoding="utf-8")
    other = w["run"] / "checkpoints" / "ckpt_step_900"
    other.mkdir()
    why = _refusal(w, TP.apply_sets(w["cmd"], [("--proprio-token-skip", "3"), ("--resume", str(other))]))
    assert why and "has no preflight record" in why, why


@pytest.mark.parametrize("sets", [[("--caregiver-record-dir", "E:/another_world_store")],
                                  [("--checkpoint-every-min", "600")], [("--minutes", "5")],
                                  [("--report-every", "50")]],
                         ids=["caregiver-record-dir", "checkpoint-every-min", "minutes", "report-every"])
def test_GAP7_every_flag_but_the_replaced_ones_reads_as_the_template(tmp_path, pin, sets):
    w = pin(_world(tmp_path))
    _write(w)
    flag = sets[0][0]
    why = _refusal(w, TP.apply_sets(w["cmd"], sets))
    assert why and "the launch's %s reads %s; the command template reads" % (flag, sets[0][1]) in why, why


def test_a_flag_of_the_template_absent_from_the_launch_is_refused(tmp_path, pin):
    """--minutes is a run flag (outside the fingerprint): without the template comparison its absence passed."""
    w = pin(_world(tmp_path))
    _write(w)
    bare = list(w["cmd"])
    i = bare.index("--minutes")
    del bare[i:i + 2]
    why = _refusal(w, bare)
    assert why and "the launch's --minutes reads absent; the command template reads 0" in why, why


def test_the_command_template_reads_its_pinned_sha256(tmp_path, pin):
    w = pin(_world(tmp_path / "written"))
    w["template"].write_text(w["template"].read_text(encoding="utf-8") + " ", encoding="utf-8")
    _refused_written(w, "the command template %s reads sha256 %s; the route pins %s" % (
        w["template"], _sha(w["template"].read_bytes()), w["template_sha"]))
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    pin(w2, command_template_sha256="5" * 64)
    _refused_at_launch_and_noted(w2, "the command template %s reads sha256 %s; the route pins %s" % (
        w2["template"], w2["template_sha"], "5" * 64))


def test_a_missing_or_malformed_command_template_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    missing = tmp_path / "no_template.json"
    pin(w, command_template=str(missing))
    _refused_written(w, "the command template %s cannot be read" % missing)
    w["template"].write_text("{}", encoding="utf-8")
    pin(w, command_template_sha256=_sha(w["template"].read_bytes()))
    _refused_written(w, "the command template %s is not a command (a JSON list)" % w["template"])


def test_the_records_configuration_flags_are_the_templates(tmp_path, pin):
    w = pin(_world(tmp_path))
    rec = _record(w)
    rec["flags"]["--intention"] = "off"
    _save_record(w, rec)
    _refused_written(w, "the record's configuration flags are not the command template's: --intention reads off, the "
                        "template on")


@pytest.mark.parametrize("sets, needle", [
    (lambda w: [("--out", str(w["run"]))], "the launch's --out %s exists already"),
    (lambda w: [("--out", str(w["pres"] / "new_out"))], "the launch's --out %s lies in the preserved folder"),
    (lambda w: [("--checkpoint-storage", str(w["pres"] / "run" / "checkpoints"))],
     "the launch's --checkpoint-storage %s lies in the preserved folder"),
], ids=["out-exists", "out-in-preservation", "storage-in-preservation"])
def test_GAP8_the_launch_writes_its_own_new_folders_outside_the_preservation(tmp_path, pin, sets, needle):
    w = pin(_world(tmp_path))
    _write(w)
    changes = sets(w)
    why = _refusal(w, TP.apply_sets(w["cmd"], changes))
    assert why and needle % changes[0][1] in why, why


def test_the_launch_names_its_out_and_checkpoint_storage(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    for flag in ("--out", "--checkpoint-storage"):
        bare = list(w["cmd"])
        i = bare.index(flag)
        del bare[i:i + 2]
        why = _refusal(w, bare)
        assert why and "the launch names no %s" % flag in why, why


def test_a_latent_output_in_the_preservation_is_refused_at_launch(tmp_path, pin):
    w = pin(_world(tmp_path))
    inside = w["pres"] / "latent_new"
    _write(w, run_flags={"--latent-body-resume": str(w["latent"]), "--latent-body-output": str(inside)})
    why = _refusal(w, TP.apply_sets(w["cmd"], [("--latent-body-output", str(inside))]))
    assert why and "the launch's --latent-body-output %s lies in the preserved folder" % inside in why, why


def test_a_changed_checkout_code_is_refused_when_written_and_when_noted(tmp_path, pin):
    needle = "the checkout's trainer code"
    w = pin(_world(tmp_path))
    (w["src"] / "lem" / "model" / "core.py").write_text("x = 2\n", encoding="utf-8")
    _refused_written(w, needle)
    w2 = pin(_world(tmp_path / "noted"))
    _write(w2)
    (w2["src"] / "lem" / "model" / "core.py").write_text("x = 2\n", encoding="utf-8")
    _refused_noted(w2, needle)


def test_a_record_whose_settings_were_edited_is_refused_when_written_at_launch_and_when_noted(tmp_path, pin):
    needle = "its settings or code were changed after it was written"
    w = pin(_world(tmp_path))
    rec = _record(w)
    rec["code_hash"] = "0" * 64
    _save_record(w, rec)
    _refused_written(w, needle)
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    rec = _record(w2)
    rec["code_hash"] = "0" * 64
    _save_record(w2, rec)
    _refused_at_launch_and_noted(w2, needle)


def test_a_record_written_to_after_the_adoption_is_refused_until_the_adoption_is_written_again(tmp_path, pin):
    needle = "changed since the adoption was written"
    w = pin(_world(tmp_path))
    _write(w)
    TP.add_state(w["fp"], str(w["ck"]), "the same checkpoint named again (a write that changes no gate)", "tester",
                 root=w["root"])
    _refused_at_launch_and_noted(w, needle)
    _write(w)
    assert _refusal(w) is None


def test_the_run_flags_the_adoption_names_are_carried_exactly(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    for flag, value in (("--latent-body-output", str(w["lat_out"]) + "_other"),
                        ("--latent-body-resume", str(w["latent"]) + ".other")):
        why = _refusal(w, TP.apply_sets(w["cmd"], [(flag, value)]))
        assert why and "the launch's %s reads %s" % (flag, value) in why and "(run_flags, compared exactly)" in why, why


def test_the_files_name_their_own_configuration(tmp_path, pin):
    w = pin(_world(tmp_path))
    rec = _record(w)
    rec["fingerprint"] = OTHER_FP
    _save_record(w, rec)
    _refused_written(w, "the record names configuration %s, not %s" % (OTHER_FP, w["fp"]))
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    held = _held(w2)
    held["fingerprint"] = OTHER_FP
    TP._save(TP.adoption_path(w2["fp"], w2["root"]), held)
    _refused_at_launch_and_noted(w2, "the adoption file names configuration %s, not %s" % (OTHER_FP, w2["fp"]))


# ------------------------------------------------------------------------------ (9) REUSED AUTHORIZATION
def _other_adoption(w, **block):
    """A hand-written adoption file of another configuration in the same preflight folder."""
    TP._save(TP.adoption_path(OTHER_FP, w["root"]), dict(fingerprint=OTHER_FP, block=dict(
        dict(authorization="other words", decision=dict(id="another-decision", message_sha256="f" * 64)), **block)))


def test_a_decision_id_already_used_is_refused_when_written_at_launch_and_when_noted(tmp_path, pin):
    _refused_everywhere(tmp_path, pin, "the decision %s already authorized the adoption" % DECISION,
                        lambda w: _other_adoption(w, decision=dict(id=DECISION, message_sha256="f" * 64)))


def test_a_decision_message_already_used_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    _other_adoption(w, decision=dict(id="another-decision", message_sha256=w["message_sha"]))
    _refused_written(w, "the decision message whose sha256 reads %s already authorized" % w["message_sha"])
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    _other_adoption(w2, decision=dict(id="another-decision", message_sha256=w2["message_sha"]))
    _refused_at_launch_and_noted(w2, "the decision message whose sha256 reads %s already authorized" % w2["message_sha"])


def _continuation_elsewhere(w, words):
    TP._save(Path(w["root"]) / ("%s.json" % OTHER_FP), dict(fingerprint=OTHER_FP, continuation=dict(
        authorization=words, authorized_at="Oct 12, 2026 7:48 PM PDT")))


def _experiment_elsewhere(w, words):
    TP._save(Path(w["root"]) / ("%s.json" % OTHER_FP), dict(fingerprint=OTHER_FP, repair_run=dict(experiment=dict(
        id="exp", authorization=words))))


def _extension_elsewhere(w, words):
    TP._save(Path(w["root"]) / TP.EXTENSIONS_DIR / ("%s.json" % OTHER_FP), dict(block=dict(authorization=words)))


def _instance_elsewhere(w, words):
    TP._save(Path(w["root"]) / TP.INSTANCES_DIR / OTHER_FP / "exp.json", dict(block=dict(experiment=dict(
        id="exp", authorization=words))))


def _adoption_elsewhere(w, words):
    _other_adoption(w, authorization=words)


WHERE = [(_continuation_elsewhere, "the continuation of record %s" % OTHER_FP),
         (_experiment_elsewhere, "experiment exp of record %s" % OTHER_FP),
         (_extension_elsewhere, "the extension"),
         (_instance_elsewhere, "the fresh instance"),
         (_adoption_elsewhere, "the adoption")]
WHERE_IDS = ["continuation", "experiment", "extension", "instance", "adoption"]


@pytest.mark.parametrize("make, where", WHERE, ids=WHERE_IDS)
def test_authorization_words_of_another_block_are_refused(tmp_path, pin, make, where):
    needle = "the adoption's authorization and the authorization words of %s" % where          # then its file
    w = pin(_world(tmp_path / "written"))
    make(w, "  " + AUTHORIZATION.replace(" ", "\n ", 3) + " ")               # whitespace aside, the same words
    _refused_written(w, needle)
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    make(w2, AUTHORIZATION)
    _refused_at_launch_and_noted(w2, needle)


def test_GAP4_another_blocks_words_inside_this_authorization_are_refused(tmp_path, pin):
    """Another block's words with the decision's id appended (and the delegation sentences) lie within this one."""
    w = pin(_world(tmp_path))
    words = "I approve D:/UEMA_Research/external/D/UEMA_scratch/body1_extension/DRAFT_AUTHORIZATION.txt (stand-in for the extension's words)"
    _extension_elsewhere(w, words)
    _refused_written(w, "the adoption's authorization and the authorization words of the extension",
                     authorization=AUTHORIZATION + "\n" + words + " " + DECISION)
    w2 = pin(_world(tmp_path / "theirs"))                                       # this one's words within theirs
    _extension_elsewhere(w2, "Preamble. " + AUTHORIZATION + " Postscript.")
    _refused_written(w2, "lie one within the other")


def test_authorization_words_that_do_not_name_the_decision_are_refused(tmp_path, pin):
    needle = "the adoption's authorization does not name its decision %s" % DECISION
    words = "\n".join('"%s"' % s for s in SENTENCES)
    w = pin(_world(tmp_path))
    _refused_written(w, needle, authorization=words)
    _write(w)
    _edit_adoption(w, authorization=words)
    _refused_at_launch_and_noted(w, needle)


def test_a_preflight_file_that_cannot_be_read_leaves_the_reuse_unchecked_and_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    bad = Path(w["root"]) / TP.EXTENSIONS_DIR / "broken.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json", encoding="utf-8")
    _refused_written(w, "%s cannot be read, so the reuse of the adoption's authorization" % bad)


# ------------------------------------------------------------------------------ (10) OUT-OF-BOUND LAUNCH
def test_the_launch_ends_exactly_at_end_step(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    for steps in ("1501", "1499", "1000"):
        why = _refusal(w, TP.apply_sets(w["cmd"], [("--max-steps", steps)]))
        assert why and "the launch is not bounded as declared: --max-steps reads %s" % steps in why, why
    bare = list(w["cmd"])
    del bare[bare.index("--max-steps"):bare.index("--max-steps") + 2]
    why = _refusal(w, bare)
    assert why and "--max-steps reads absent" in why, why


def test_max_model_steps_other_than_end_minus_start_is_refused(tmp_path, pin):
    needle = "max_model_steps 499 is not its end_step minus its start_step (1500 - 1000 = 500)"
    w = pin(_world(tmp_path))
    _refused_written(w, needle, max_model_steps=499)
    _write(w)
    _edit_adoption(w, max_model_steps=499)
    _refused_at_launch_and_noted(w, needle)


def test_an_adoption_without_a_bound_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path), end=1000)
    _refused_written(w, "the adoption's duration is not bounded", end_step=1000, max_model_steps=0)


@pytest.mark.parametrize("cap", [0, -1.0, "20683", float("nan"), True],
                         ids=["zero", "negative", "text", "nan", "bool"])
def test_the_wall_cap_is_a_number_of_seconds_above_zero(tmp_path, pin, cap):
    _refused_written(pin(_world(tmp_path)), "wall_cap_s must be a number of seconds above 0", wall_cap_s=cap)


# ------------------------------------------------------------------------------ (11) THE GATES AND THE BLOCK'S FORM
def test_GAP1_a_gate_judged_PASS_or_NA_refuses_the_adoption(tmp_path, pin):
    """The decision: "Keep each gate FAIL". A gate judged PASS (with any written evidence) or N/A is refused."""
    for status in ("PASS", "N/A"):
        w = pin(_world(tmp_path / status.replace("/", "")))
        TP.set_gate(w["fp"], 7, status, "claimed without a run", ["any text"], "tester", root=w["root"])
        _refused_written(w, "gate 7 End-to-end causal dry run reads %s; an adoption launches only with every gate FAIL"
                         % status, exceptions=_exceptions([1, 2, 3, 4, 5, 6]))
    w = pin(_world(tmp_path / "launch"))
    _write(w)
    TP.set_gate(w["fp"], 7, "PASS", "claimed after the adoption was written", ["any text"], "tester", root=w["root"])
    why = _refusal(w)
    assert why and "gate 7 End-to-end causal dry run reads PASS; an adoption launches only with every gate FAIL" in why


def test_GAP1b_all_gates_PASS_neither_admit_the_adoption_nor_normal_training_after_its_use(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    _mark(w, _folder(w, "run_used"))
    for n in range(1, 8):
        TP.set_gate(w["fp"], n, "PASS", "claimed", ["any text"], "tester", root=w["root"])
    long_run = TP.apply_sets(w["cmd"], [("--max-steps", "999999")])
    why = _refusal(w, long_run)
    assert why and LISTED in why, why                       # normal training from the preserved checkpoint: refused
    w2 = pin(_world(tmp_path / "written"))
    for n in range(1, 8):
        TP.set_gate(w2["fp"], n, "PASS", "claimed", ["any text"], "tester", root=w2["root"])
    _refused_written(w2, "reads PASS; an adoption launches only with every gate FAIL", exceptions={})


def test_a_failing_gate_without_its_complete_exception_or_a_gate_never_judged_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    _refused_written(w, "gate 5 Causal controllability reads FAIL and the adoption names no complete exception",
                     exceptions=_exceptions([1, 2, 3, 4, 6, 7]))
    no_relevance = dict(_exceptions(), **{"3": {k: v for k, v in _exceptions()["3"].items() if k != "relevance"}})
    _refused_written(w, "gate 3 Scale/conditioning reads FAIL and the adoption names no complete exception",
                     exceptions=no_relevance)
    rec = _record(w)
    rec["gates"] = [dict(g, status="OPEN") if g["n"] == 2 else g for g in rec["gates"]]
    _save_record(w, rec)
    _refused_written(w, "gate 2 Gradient flow reads OPEN")
    rec["gates"] = [g for g in rec["gates"] if g["n"] != 2]
    _save_record(w, rec)
    _refused_written(w, "gate 2 Gradient flow reads OPEN (missing from the record)")


def test_a_fail_gate_without_its_reason_and_evidence_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    rec = _record(w)
    rec["gates"] = [dict(g, reason="", evidence=[]) if g["n"] == 6 else g for g in rec["gates"]]
    _save_record(w, rec)
    _refused_written(w, "gate 6 Objective/test alignment reads FAIL without its reason and evidence")


def test_a_gate_judged_again_after_the_adoption_was_written_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    time.sleep(0.02)
    TP.set_gate(w["fp"], 4, "FAIL", "judged again", ["row_4b"], "tester", root=w["root"])
    why = _refusal(w)
    assert why and "gate 4 Local stability was judged again since its exception was written" in why, why
    _write(w)
    assert _refusal(w) is None


def test_an_adoption_file_without_its_written_time_is_refused_at_launch(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    held = _held(w)
    held["block"].pop("written_at_s")
    TP._save(TP.adoption_path(w["fp"], w["root"]), held)
    why = _refusal(w)
    assert why and "gate 1 Information flow: its exception cannot be tied to the failure it excuses" in why, why


def test_an_exception_never_reads_a_status_other_than_fail(tmp_path, pin):
    w = pin(_world(tmp_path))
    claims = dict(_exceptions(), **{"5": dict(_exceptions()["5"], status="PASS")})
    _refused_written(w, "the adoption's exception for gate 5 reads status PASS: an exception leaves its gate FAIL",
                     exceptions=claims)
    assert [g["status"] for g in _record(w)["gates"]] == ["FAIL"] * 7


@pytest.mark.parametrize("exceptions, needle", [
    (dict(_exceptions(), **{"5": dict(_exceptions()["5"], evidence=[])}), "exception for gate 5 lacks its evidence"),
    (dict(_exceptions(), **{"5": dict(_exceptions()["5"], relevance=" ")}), "exception for gate 5 lacks its relevance"),
    (dict(_exceptions(), x=dict(_exceptions()["1"])), "the adoption's exception 'x' is not a gate number"),
    (dict(_exceptions(), **{"9": dict(_exceptions()["1"])}), "the adoption's exception 9 is not one of the seven gates"),
    (dict(_exceptions(), **{"5": "fails"}), "the adoption's exception for gate 5 must be {failure"),
], ids=["evidence", "relevance", "x", "nine", "notdict"])
def test_the_exceptions_form(tmp_path, pin, exceptions, needle):
    _refused_written(pin(_world(tmp_path)), needle, exceptions=exceptions)


@pytest.mark.parametrize("key", TP.ADOPTION_FIELDS)
def test_every_field_is_required(tmp_path, pin, key):
    w = pin(_world(tmp_path))
    with pytest.raises(TP.PreflightError, match="the adoption's %s is missing" % key.replace("_", " ")):
        _write_without(w, key)
    for empty in ("", "   \n "):
        if key in TP.ADOPTION_TEXT_FIELDS:
            with pytest.raises(TP.PreflightError, match="the adoption's %s is missing" % key.replace("_", " ")):
                _write(w, **{key: empty})


def test_the_exceptions_are_a_block(tmp_path, pin):
    w = pin(_world(tmp_path))
    with pytest.raises(TP.PreflightError, match="the adoption's exceptions must be a block"):
        _write_without(w, "exceptions")
    with pytest.raises(TP.PreflightError, match="the adoption's exceptions must be a block"):
        _write(w, exceptions=[["1", "x"]])


@pytest.mark.parametrize("key, value", [("authorization", ["words"]), ("authorized_by", {"who": "a name"}),
                                        ("authorized_at", 1791880000.0), ("purpose", 12),
                                        ("stop_lines_row", ["adopt_stop_row"])],
                         ids=["authorization", "authorized_by", "authorized_at", "purpose", "stop_lines_row"])
def test_the_named_fields_must_be_text(tmp_path, pin, key, value):
    _refused_written(pin(_world(tmp_path)), "the adoption's %s must be text" % key.replace("_", " "), **{key: value})


SUB_CASES = [
    (lambda w: dict(predecessor=_pred(w, run=None)), "the adoption's predecessor run must be text"),
    (lambda w: dict(predecessor=_pred(w, model_sha256="ABC")), "the adoption's predecessor model_sha256 must be a sha256"),
    (lambda w: dict(predecessor=_pred(w, outcome_receipt_sha256="z" * 64)),
     "the adoption's predecessor outcome_receipt_sha256 must be a sha256"),
    (lambda w: dict(decision=dict(id=DECISION, message_sha256=w["message_sha"])),
     "the adoption's decision author must be text"),
    (lambda w: dict(decision=dict(id=DECISION, message_sha256="1234", author=AUTHOR)),
     "the adoption's decision message_sha256 must be a sha256"),
    (lambda w: dict(code_transition=dict(from_commit=FROM_COMMIT, to_commit="b2", commits="b2")),
     "names its commits as a list of text"),
    (lambda w: dict(code_transition=dict(from_commit=FROM_COMMIT, to_commit="b2", commits=[])),
     "names its commits as a list of text"),
    (lambda w: dict(code_transition=dict(from_commit=FROM_COMMIT, commits=["b2"])),
     "the adoption's code transition to_commit"),
    (lambda w: dict(code_transition=["a1", "b2"]), "the adoption's code transition must be a block"),
    (lambda w: dict(code_transition=dict(from_commit=FROM_COMMIT, to_commit=FROM_COMMIT, commits=[FROM_COMMIT])),
     "as both its from_commit and its to_commit (%s)" % FROM_COMMIT),
]


@pytest.mark.parametrize("change, needle", SUB_CASES, ids=[
    "pred-run", "model-sha", "receipt-sha", "author", "message-sha", "commits-text", "commits-empty", "to-commit",
    "ct-block", "same-commit"])
def test_the_predecessor_code_transition_and_decision_name_their_fields(tmp_path, pin, change, needle):
    w = pin(_world(tmp_path))
    _refused_written(w, needle, **change(w))


@pytest.mark.parametrize("more, needle", [(dict(role="continuation"), "the adoption's role reads continuation"),
                                          (dict(repeats=2), "an adoption is used once"),
                                          (dict(repeats=True), "an adoption is used once"),
                                          (dict(target_gates=[5]), "an adoption targets no gate"),
                                          (dict(start_step="1000"), "start_step must be a whole number"),
                                          (dict(end_step=1500.0), "end_step must be a whole number"),
                                          (dict(max_model_steps="500"), "max_model_steps must be a whole number")],
                         ids=["role", "repeats2", "repeatsTrue", "targets", "start-str", "end-float", "max-str"])
def test_role_repeats_targets_and_whole_numbers(tmp_path, pin, more, needle):
    _refused_written(pin(_world(tmp_path)), needle, **more)


@pytest.mark.parametrize("key, what", [("repair_run", "repair run"), ("continuation", "continuation")],
                         ids=["repair_run", "continuation"])
def test_a_record_with_another_bounded_route_is_refused(tmp_path, pin, key, what):
    w = pin(_world(tmp_path))
    rec = _record(w)
    rec[key] = dict(start_checkpoint=str(w["run"] / "checkpoints" / "ckpt_step_900"), launched_at="earlier",
                    launched_at_s=1.0, run=str(w["run"]))
    _save_record(w, rec)
    _refused_written(w, "the record holds a %s" % what)


def test_a_start_that_is_not_a_model_state_of_the_record_is_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    rec = _record(w)
    rec["states"] = [dict(checkpoint=str(w["run"] / "checkpoints" / "ckpt_step_900"), at="earlier", at_s=1.0)]
    _save_record(w, rec)
    _refused_written(w, "is not a model state the record names")


def test_the_stop_criteria_are_checked_as_a_continuations(tmp_path, pin):
    w = pin(_world(tmp_path))
    _refused_written(w, "no stop criterion has mode stop", stop_criteria=[dict(STOP_CRITERIA[0], mode="alert")])
    _refused_written(w, "reads no number", stop_criteria=[dict(STOP_CRITERIA[0], path="causal_pe_per_modality.visoin")])
    # at the launch's own report rate: with --report-every 1000 the first full stats record comes 1,000 model steps in
    w2 = pin(_world(tmp_path / "rate"))
    _write(w2)
    w2["template"].write_text(w2["template"].read_text(encoding="utf-8").replace('"100"', '"1000"'), encoding="utf-8")
    pin(w2, command_template_sha256=_sha(w2["template"].read_bytes()))
    why = _refusal(w2, TP.apply_sets(w2["cmd"], [("--report-every", "1000")]))
    assert why and "the adoption ends 500 model steps after its start, before the watcher first judges" in why, why


def test_the_run_flags_name_both_latent_body_flags_an_existing_state_and_a_new_folder(tmp_path, pin):
    w = pin(_world(tmp_path))
    _refused_written(w, "the adoption's run_flags name --latent-body-output",
                     run_flags={"--latent-body-resume": str(w["latent"])})
    _refused_written(w, "the adoption's run_flags name --latent-body-resume",
                     run_flags={"--latent-body-output": str(w["lat_out"])})
    _refused_written(w, "run flag --intention", run_flags={"--latent-body-resume": str(w["latent"]),
                                                          "--latent-body-output": str(w["lat_out"]),
                                                          "--intention": "on"})
    _refused_written(w, "is not a file", run_flags={"--latent-body-resume": str(w["latent"]) + ".missing",
                                                   "--latent-body-output": str(w["lat_out"])})
    w["lat_out"].mkdir()
    _refused_written(w, "exists already; an adoption writes a new folder")
    w2 = pin(_world(tmp_path / "launch"))
    _write(w2)
    w2["lat_out"].mkdir()
    why = _refusal(w2)
    assert why and "the latent body output folder %s exists already" % w2["lat_out"] in why, why
    _note(w2, _folder(w2, "run_started"))                               # noted after the trainer made its folder: kept


def test_set_adoption_needs_a_record_and_who_writes_it(tmp_path, pin):
    w = pin(_world(tmp_path))
    with pytest.raises(TP.PreflightError, match="no record for configuration"):
        TP.set_adoption(OTHER_FP, _block(w), "tester", root=w["root"], registry=w["reg"], src=w["src"])
    with pytest.raises(TP.PreflightError, match="who writes it"):
        TP.set_adoption(w["fp"], _block(w), " ", root=w["root"], registry=w["reg"], src=w["src"])


# ------------------------------------------------------------------------------ (12) THE REGISTRY ROWS
@pytest.mark.parametrize("more, needle", [
    (dict(stop_lines_row="missing_row"), "the training row missing_row is not in the test registry"),
    (dict(completion_jobs_row="missing_row"), "the completion jobs' row missing_row is not in the test registry"),
    (dict(stop_lines_row="done_row"), "the training row done_row reads status done; an adoption's row"),
    (dict(completion_jobs_row="running_row"), "the completion jobs' row running_row reads status running"),
], ids=["stop-missing", "done-missing", "stop-done", "done-running"])
def test_the_registry_rows_are_declared_and_not_yet_run(tmp_path, pin, more, needle):
    w = pin(_world(tmp_path))
    _refused_written(w, needle, **more)
    _write(w)
    _edit_adoption(w, **more)
    why = _refusal(w)
    assert why and needle in why, why


def test_a_row_reset_by_hand_after_the_launch_is_still_refused(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    run = _folder(w, "run_first")
    _note(w, run)
    _stats(run, ((1100, 108.0),))                                       # the attempt trained: the one use
    _edit_adoption(w, attempts=[])                                      # a hand edit of the adoption file
    _edit_row(w, "adopt_stop_row", status="failed")                     # the run owner's end of the run row
    why = _refusal(w)
    assert why and "the training row adopt_stop_row reads status failed" in why, why
    _edit_row(w, "adopt_stop_row", status="queued")                     # a hand reset of the run row too
    why = _refusal(w)
    assert why and ATTEMPT in why, why
    with pytest.raises(TP.PreflightError, match=ATTEMPT):
        _note(w, _folder(w, "run_second"))


@pytest.mark.parametrize("row, waits_on, allowed", [
    ("adopt_done_row", "the run row adopt_stop_row ends", True),
    ("adopt_done_row", "the end of proxy decision %s's run" % DECISION, True),
    ("adopt_done_row", "the operator", False),
    ("adopt_stop_row", "the run row adopt_stop_row ends", False),
], ids=["completion-waits-on-run", "completion-waits-on-decision", "completion-other", "run-row"])
def test_only_the_completion_row_may_be_blocked_waiting_on_this_adoption(tmp_path, pin, row, waits_on, allowed):
    w = pin(_world(tmp_path))
    _edit_row(w, row, blocked=dict(reason="waits", waits_on=waits_on, by="tester", at="now"))
    if allowed:
        _write(w)
        assert _refusal(w) is None
        _note(w, _folder(w, "run_a"))
    else:
        _refused_written(w, "is blocked (waits; it waits on %s)" % waits_on)


# ------------------------------------------------------------------------------ (13) the noting needs src; the command line
def test_a_row_that_cannot_be_written_refuses_the_noting(tmp_path, pin):
    """Two rows with the training row's id: test_registry.update_row writes no row of an id held twice."""
    w = pin(_world(tmp_path))
    _write(w)
    data = json.loads(w["reg"].read_text(encoding="utf-8"))
    data["rows"].append(dict(id="adopt_stop_row", status="written"))
    w["reg"].write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(TP.PreflightError, match="could not be recorded: no single row of .* has id adopt_stop_row"):
        _note(w, _folder(w, "run_a"))
    assert not _held(w)["block"].get("attempts")


def test_the_noting_of_an_adoption_needs_src(tmp_path, pin):
    w = pin(_world(tmp_path))
    _write(w)
    run = _folder(w, "run_a")
    for fn in (TP.note_attempt, TP.mark_launched):
        with pytest.raises(TP.PreflightError, match="noted with src"):
            fn(w["fp"], str(run), "launcher", root=w["root"], block=TP.ADOPTION, registry=w["reg"])


def test_without_a_registry_the_noting_is_refused_and_writes_nothing(tmp_path, pin):
    """The fixture no_production_registry: the default registry is a missing file, so a call without a registry names
    no row (and production's registry is never read or written by these tests)."""
    w = pin(_world(tmp_path))
    _write(w)
    with pytest.raises(TP.PreflightError, match="is not in the test registry"):
        TP.note_attempt(w["fp"], str(_folder(w, "run_a")), "launcher", root=w["root"], block=TP.ADOPTION, src=w["src"])
    assert not _held(w)["block"].get("attempts")


def test_the_command_line_writes_an_adoption_and_marks_its_launch(tmp_path, pin, monkeypatch, capsys):
    w = pin(_world(tmp_path))
    set0 = TP.set_adoption
    monkeypatch.setattr(TP, "set_adoption", lambda f, b, by, *x, **k: set0(f, b, by, root=w["root"], registry=w["reg"],
                                                                          src=w["src"]))
    mark0 = TP.mark_launched
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: mark0(f, r, by, root=w["root"], **k))
    path = tmp_path / "adoption_block.json"
    path.write_text(json.dumps(_block(w)), encoding="utf-8-sig")
    assert TP.main(["adoption", w["fp"], "--by", "tester"]) == 2
    assert "adoption needs --block-file" in capsys.readouterr().out
    assert TP.main(["adoption", w["fp"], "--block-file", str(path), "--by", "tester"]) == 0
    assert "adoption written" in capsys.readouterr().out
    run = _folder(w, "run_cli")
    assert TP.main(["launched", w["fp"], "--run", str(run), "--adoption", "--extension", "--by", "t"]) == 2
    assert "launched marks one block" in capsys.readouterr().out
    assert TP.main(["launched", w["fp"], "--run", str(run), "--adoption", "--registry", str(w["reg"]),
                    "--by", "tester"]) == 2
    assert "noted with src" in capsys.readouterr().out
    assert TP.main(["launched", w["fp"], "--run", str(run), "--adoption", "--src", str(w["src"]), "--registry",
                    str(w["reg"]), "--by", "tester"]) == 0
    assert "adoption marked launched" in capsys.readouterr().out
    assert _held(w)["block"]["run"] == str(run.resolve())
    assert _rows(w)["adopt_stop_row"][TP.ADOPTION_ROW_FIELD]["launched_at"]
    assert TP.main(["launched", w["fp"], "--run", str(run), "--adoption", "--src", str(w["src"]), "--registry",
                    str(w["reg"]), "--by", "tester"]) == 2
    assert "launched already" in capsys.readouterr().out


# ------------------------------------------------------------------------------ (14) THE PAIR'S SIBLING ADOPTIONS
# a2-admission-repair-authorized-20261023, item 1: two configurations, each with its OWN bound route at the same
# preserved checkpoint and its OWN unused adoption, are each judged by their own adoption's full conditions; a third
# configuration borrows nothing; a missing, malformed, stale, invalid or used own adoption is refused; a changed command
# or checkpoint is refused; a used sibling frees the other and cannot relaunch itself; one block per decision still holds.
DECISION_B = "proxy-adopt-test-sibling-20261023"
AUTHORIZATION_B = ("\n".join('"%s"' % s for s in SENTENCES) + "\nDelegation reference: the operator's delegation of "
                   "Oct 13, 2026; proxy decision %s, by %s." % (DECISION_B, AUTHOR))
MESSAGE_TEXT_B = "H84 PROXY DECISION: %s\n(a stand-in for the sibling decision's text)\n" % DECISION_B
SIBLING = "BLOCKED: adoption of configuration"


def _pair(tmp_path, monkeypatch, write_b=True):
    """The world's configuration A and a sibling B (--proprio-token-skip 3, its own folders) from the same preserved
    checkpoint, each with its own bound route, decision, command template, rows and adoption."""
    w = _world(tmp_path)
    out_b, lat_b = tmp_path / "new_run_b", tmp_path / "lat_out_b"
    cmd_b = TP.apply_sets(w["cmd"], [("--proprio-token-skip", "3"), ("--out", str(out_b)),
                                     ("--checkpoint-storage", str(out_b / "checkpoints")),
                                     ("--latent-body-output", str(lat_b))])
    fp_b = Path(TP.new_record(cmd_b, "the sibling configuration", "tester", root=w["root"], src=w["src"])).stem
    for n in range(1, 8):
        TP.set_gate(fp_b, n, "FAIL", "fails on the preserved checkpoint", ["row_%d" % n], "tester", root=w["root"])
    tpl = json.loads(w["template"].read_text(encoding="utf-8"))
    tpl[tpl.index("--proprio-token-skip") + 1] = "3"
    template_b = tmp_path / "command_template_b.json"
    template_b.write_text(json.dumps(tpl, indent=1), encoding="utf-8")
    message_b = tmp_path / "message_b.json"
    message_sha_b = _sha(MESSAGE_TEXT_B.encode("utf-8"))
    message_b.write_text(json.dumps(dict(id=DECISION_B, text=MESSAGE_TEXT_B, sha256=message_sha_b)), encoding="utf-8")
    data = json.loads(w["reg"].read_text(encoding="utf-8"))
    data["rows"] += [dict(id="adopt_stop_row_b", status="written"), dict(id="adopt_done_row_b", status="queued")]
    w["reg"].write_text(json.dumps(data), encoding="utf-8")
    route_b = _route(w, fingerprint=fp_b, decision=DECISION_B, decision_message=str(message_b),
                     decision_text_sha256=message_sha_b, command_template=str(template_b),
                     command_template_sha256=_sha(template_b.read_bytes()), run_row="adopt_stop_row_b")
    monkeypatch.setattr(TP, "ADOPTION_ROUTES", (_route(w), route_b))
    _write(w)
    b = dict(fp=fp_b, cmd=cmd_b, block=_block(
        w, decision=dict(id=DECISION_B, message_sha256=message_sha_b, author=AUTHOR), authorization=AUTHORIZATION_B,
        stop_lines_row="adopt_stop_row_b", completion_jobs_row="adopt_done_row_b",
        run_flags={"--latent-body-resume": str(w["latent"]), "--latent-body-output": str(lat_b)}))
    if write_b:
        TP.set_adoption(fp_b, b["block"], "tester", root=w["root"], registry=w["reg"], src=w["src"])
    return w, b


def _edit_block(w, fp, block):
    path = TP.adoption_path(fp, w["root"])
    held = json.loads(path.read_text(encoding="utf-8"))
    held["block"] = block(held["block"]) if callable(block) else block
    TP._save(path, held)


def test_pair_each_sibling_is_admitted_by_its_own_adoption(tmp_path, monkeypatch):
    w, b = _pair(tmp_path, monkeypatch)
    assert _refusal(w) is None
    assert _refusal(w, b["cmd"]) is None
    assert (_plan(w)["block"], _plan(w)["fingerprint"]) == ("adoption", w["fp"])
    assert (_plan(w, b["cmd"])["block"], _plan(w, b["cmd"])["fingerprint"]) == ("adoption", b["fp"])


def test_pair_a_third_configuration_borrows_nothing(tmp_path, monkeypatch):
    w, b = _pair(tmp_path, monkeypatch)
    third = TP.apply_sets(w["cmd"], [("--proprio-token-skip", "1")])
    fp_c = Path(TP.new_record(third, "a third configuration", "tester", root=w["root"], src=w["src"])).stem
    for n in range(1, 8):
        TP.set_gate(fp_c, n, "PASS", "stand-in", ["e"], "tester", root=w["root"])
    why = _refusal(w, third)
    assert why and SIBLING in why and "is not the adoption's" in why, why


def test_pair_a_missing_own_adoption_keeps_the_sibling_refusal(tmp_path, monkeypatch):
    w, b = _pair(tmp_path, monkeypatch, write_b=False)
    why = _refusal(w, b["cmd"])
    assert why and "%s %s" % (SIBLING, w["fp"]) in why and "is not the adoption's" in why, why
    assert _refusal(w) is None                        # B has no adoption: nothing blocks A


def test_pair_a_malformed_own_adoption_is_refused(tmp_path, monkeypatch):
    for case, block in (("partial", lambda ab: dict(start_checkpoint=ab["start_checkpoint"])), ("not_a_dict", "x")):
        w, b = _pair(tmp_path / case, monkeypatch)
        _edit_block(w, b["fp"], block)
        why = _refusal(w, b["cmd"])
        assert why and why.startswith("BLOCKED"), (case, why)


def test_pair_an_invalid_or_stale_own_adoption_is_refused_by_its_own_conditions(tmp_path, monkeypatch):
    w, b = _pair(tmp_path / "invalid", monkeypatch)
    _edit_block(w, b["fp"], lambda ab: dict(ab, end_step=1400))
    why = _refusal(w, b["cmd"])
    assert why and "%s %s" % (SIBLING, b["fp"]) in why and "condition(s) unmet" in why, why
    w2, b2 = _pair(tmp_path / "stale", monkeypatch)
    rec = TP._load(TP.record_path(b2["fp"], w2["root"]))
    rec["edited_after_the_adoption"] = True
    TP._save(TP.record_path(b2["fp"], w2["root"]), rec)
    why = _refusal(w2, b2["cmd"])
    assert why and "%s %s" % (SIBLING, b2["fp"]) in why and "condition(s) unmet" in why, why


def test_pair_a_used_sibling_frees_the_other_and_cannot_relaunch_itself(tmp_path, monkeypatch):
    w, b = _pair(tmp_path, monkeypatch)
    run = _folder(w, "run_a")
    _note(w, run)
    _edit_row(w, "adopt_stop_row", status="running")
    _mark(w, run)                                     # A's adoption is used
    assert _refusal(w, b["cmd"]) is None              # B: judged by its own adoption alone
    why = _refusal(w)                                 # A: its used adoption gives nothing; B's unused one refuses it
    assert why and "%s %s" % (SIBLING, b["fp"]) in why, why


def test_pair_a_changed_command_or_checkpoint_is_refused(tmp_path, monkeypatch):
    w, b = _pair(tmp_path, monkeypatch)
    for change in ([("--proprio-token-skip", "2")], [("--max-steps", "1600")]):
        why = _refusal(w, TP.apply_sets(b["cmd"], change))
        assert why and why.startswith("BLOCKED"), (change, why)
    other = w["run"] / "checkpoints" / "ckpt_step_900"
    other.mkdir()
    why = _refusal(w, TP.apply_sets(b["cmd"], [("--resume", str(other))]))
    assert why and why.startswith("BLOCKED"), why


def test_pair_a_launch_the_mismatch_does_not_refuse_never_reaches_the_sibling_exception(tmp_path, pin, monkeypatch):
    w = pin(_world(tmp_path))
    _write(w)

    def unreachable(*_args, **_kwargs):
        raise AssertionError("_own_adoption_at was called for a launch _adoption_mismatch does not refuse")
    monkeypatch.setattr(TP, "_own_adoption_at", unreachable)
    assert _refusal(w) is None


def test_pair_one_block_per_decision_still_holds(tmp_path, monkeypatch):
    w, b = _pair(tmp_path, monkeypatch, write_b=False)
    with pytest.raises(TP.PreflightError) as exc:
        TP.set_adoption(b["fp"], dict(b["block"], decision=dict(id=DECISION, message_sha256=w["message_sha"],
                                                                author=AUTHOR), authorization=AUTHORIZATION),
                        "tester", root=w["root"], registry=w["reg"], src=w["src"])
    assert "a decision authorizes one block" in str(exc.value), str(exc.value)
