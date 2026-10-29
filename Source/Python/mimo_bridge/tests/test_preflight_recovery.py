# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE LINKED RECOVERY OF A USED ADOPTION (training_preflight.py, the module text; proxy decision
proxy-adoption-recovery-489625-20261013 of Oct 13, 2026, item 7: "Register one
linked recovery/extension naming that used record, actual stopped checkpoint, this decision, exact code/settings,
endpoint and remaining wall budget ... Do not erase history, admit a fresh unrelated adoption, manufacture a successful
predecessor or present inherited failed gates as newly measured passes ... Use existing route checks for wrong
parent/checkpoint, changed training configuration, reused authorization and excess steps/time").

The world of these tests mirrors the decision on stand-ins, in pytest's temporary folder: a preservation of the adoption's
start (model step 1,000); a new configuration's record judged FAIL on all seven gates; its adoption (1,000 to 2,000, wall
cap 2,000 s), written, noted and marked launched for a run folder that then trained to model step 1,200 and stopped (its
args.json, stats, stage_done.json, end checkpoint and latent body folder); the adoption's rows ended as the run owner ends
them (training row failed, its runner's state STOPPED; completion row failed); a preservation of the stopped run's end
(its MANIFEST naming the run's checkpoint at 1,200 as its source, the worlds companion, the preserved run's args.json and
stage_done.json); the run owner's stopped receipt, the INCOMPLETE outcome receipt and its log (the trainer's process
created at 1,000.25 s, its end observed at 1,100.75 s); the recovery decision's message; the adoption's launch command
(the recovery's template). The recovery: one bounded run from the preserved 1,200 to 2,000 within floor(2000 - 100.5) =
1,899 s. Nothing here writes a production file, and no test reads the production registry (the fixture
`no_production_registry`), Saved/ or D:/UEMA_Research/external/D/UEMA_preserved. One test reads the preflight folder of the checkout that holds
the tool (production's, in main): the command line's check without --root reads that folder's reference configuration
and finds no record of the stand-in configuration there, which shows that check's default folder is production's (the
independent review of Oct 13, 2026, finding m8). The test of the production pins reads only this module's constants.

What is held (each refusal checked when the recovery is written, at launch, and when an attempt or a launch is noted,
wherever that moment can see the condition):
  (1) the recovery admits exactly its launch; the record, the used adoption's file and the adoption's rows stay byte for
      byte as they were; the launch is recorded in the recovery's own row; used once;
  (2) WRONG PARENT: the used adoption's file, its use, its run, its decision, the record it was written against, the
      lineage, the adoption's rows, another configuration;
  (3) WRONG CHECKPOINT: the original (not preserved) end, the adoption's start, the model, the latent body state, the
      worlds companion, the MANIFEST's source, step and verdict, the preserved args.json, a run that reached its end, the
      stopped receipt, an outcome that is not INCOMPLETE;
  (4) CHANGED TRAINING CONFIGURATION: a flag, the checkout's code, a template flag, the stop criteria, the template, the
      code's commit and hash;
  (5) REUSED AUTHORIZATION: the adoption's decision, its words, its time, another recovery's decision, the decision's
      message and the delegation;
  (6) EXCESS STEPS OR TIME: --max-steps, the end, the bound, the wall cap and the receipts it is computed from;
  (7) THE GATES: the adoption's exceptions carried unchanged with their applicability; no PASS;
  (8) THE ROWS AND THE LAUNCH: the pinned training row, the completion row, a noted attempt, the launch's folders, the
      run flags, src for the noting; the command line.
"""
import hashlib
import json
import math
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
ROWS = dict(adopt_stop_row="written", adopt_done_row="queued", recover_stop_row="written", recover_done_row="queued",
            done_row="done", running_row="running", other_row="written")
DECISION_A = "proxy-adopt-test-20261013"
DECISION_R = "proxy-recovery-test-20261013"
AUTHOR = "the overseer"
SENTENCES = TP.RECOVERY_ROUTES[0]["delegation_words"]         # the operator's quoted sentences, as the route pins them
QUOTED = "\n".join('"%s"' % s for s in SENTENCES)
AUTH_A = QUOTED + ("\nDelegation reference: the operator's delegation of Oct 13, 2026; proxy decision %s, by %s: adopt "
                   "the preserved start." % (DECISION_A, AUTHOR))
AUTH_R = QUOTED + ("\nDelegation reference: the operator's delegation of Oct 13, 2026; proxy decision %s, by %s: resume "
                   "the stopped run's preserved end to the adoption's end." % (DECISION_R, AUTHOR))
MESSAGE_A = "H84 PROXY DECISION: %s\n(a stand-in for the adoption decision's text)\n" % DECISION_A
MESSAGE_R = "H84 PROXY DECISION: %s\n(a stand-in for the recovery decision's text)\n" % DECISION_R
FROM_COMMIT = "abcdef0123456789abcdef0123456789abcdef01"
TO_COMMIT = "2222222bbbb2222222bbbb2222222bbbb2222222"
START_A, STOP, END = 1000, 1200, 2000
WALL_A = 2000
T_CREATED, T_ENDED = 1000.25, 1100.75
WALL_R = int(math.floor(WALL_A - (T_ENDED - T_CREATED)))          # 1899
OTHER_FP = "0123456789abcdef"
CK1_REL = "run/checkpoints/ckpt_step_1000/"
CK2_REL = "run/checkpoints/ckpt_step_1200/"
COMPANION_REL = "episode_companion/ckpt_step_1200_4567.pt.gz"
PRESERVED_A = {CK1_REL + "meta.json": json.dumps(dict(step=1000)).encode("utf-8"),
               CK1_REL + "model.safetensors": b"the predecessor's weights at model step 1000",
               CK1_REL + "optimizer.pt": b"the predecessor's optimizer state",
               "latent_body/latest.pt": b"the predecessor's latent body state",
               "episode_companion/ckpt_step_1000_0123.pt.gz": b"the predecessor's eight worlds"}
PRESERVED_R = {CK2_REL + "meta.json": json.dumps(dict(step=1200)).encode("utf-8"),
               CK2_REL + "model.safetensors": b"the stopped run's weights at model step 1200",
               CK2_REL + "optimizer.pt": b"the stopped run's optimizer state",
               "latent_body/latest.pt": b"the stopped run's latent body state",
               COMPANION_REL: b"the stopped run's eight worlds"}
PRODUCTION_ROUTE = dict(
    fingerprint="eea1e8bc7f446fb6",
    adoption_sha256="1d15cca55d5b85ea1f87079f753a4648636c3ba18d4d5600ddf06d99ff572280",
    adoption_decision="proxy-adopt-b1e-one-run-20261013",
    adoption_run="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/run/B1A_skip13_489350",
    adoption_row="b1_adopt_B1A_skip13_489350", adoption_start=489350, original_wall_cap_s=20683,
    record_sha256="e282bfc08fb4142b9b11780115898ebc85f526c466be40ad5b9da8c22e57d083",
    checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/run/checkpoints/ckpt_step_489625",
    manifest="D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/MANIFEST.json",
    manifest_sha256="85f8f712d17f5f08d2db76993c84928e936b1fbfcb803d351d204b02da2a4a43",
    preserved_root="D:/UEMA_Research/external/D/UEMA_preserved", predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/run",
    model_sha256="b274e85cf4fca6009e10e3dd993e39f316640132204341099581a9e6e505ba2b",
    companion_sha256="984482cc4515d4c3b6c2eee38af328765699a1dd38465a4e6e3ea417ae6b8b0a",
    latent_sha256="238f2a316489da17530885f9f963c7a3fb929a1ccd52521c28cba62ac7928d1b",
    stopped_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/receipts/adoption/arm_end.json",
    stopped_receipt_sha256="b21379f67f277dcd31bb5738487559f1beffc65effe48478831731b9e87a8d80",
    outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/reading/outcome_receipt.json",
    outcome_receipt_sha256="c9e0c06eb42a1362a6c96594b83907672f9c2abf67d5177c355bd2274d800f2f",
    decision="proxy-adoption-recovery-489625-20261013",
    decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                     "proxy-adoption-recovery-489625-20261013/message.json",
    decision_text_sha256="f382f32b0240a3ba4f9fea41c7e7b7280c98dc6d32b5d60fb23e0f0fd2e1a78f",
    decision_author="the overseer",
    delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                      "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
    delegation_words=("The operator's written delegation of October 13, 2026 authorizes this route.",
                      "You will determin launch conditions in my absense!"),
    code_commit="641c931a67edf2988ec886ed15c93c376cf18f46",
    command_template="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/command.json",
    command_template_sha256="e84a386dea90ebf9612cd23220d0e7f99d2579c5e49415905536122b305208b6",
    run_row="b1_recover_B1R_skip13_489625", start=489625, end=494025, wall_cap_s=19931,
    owner_log="D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/receipts/owner.jsonl",
    trainer_created_s=1791913178.843792, trainer_exit_observed_s=1791913930.4329011)


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


def _write_files(folder: Path, files: dict):
    for rel, data in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(data)


def _manifest(pres: Path, files: dict, run_files=("stage_done.json",), **fields) -> bytes:
    """The preservation's MANIFEST.json over `files` ({relative path: bytes}) and the preserved run's `run_files` as they
    read on disk, as the preservation tool writes it ({files: [{path, bytes, sha256}], ...})."""
    files = dict(files)
    for name in run_files:
        p = pres / "run" / name
        if p.is_file():
            files["run/" + name] = p.read_bytes()
    data = dict(fields, verdict=fields.get("verdict", "pass"),
                files=[dict(path=rel, bytes=len(b), sha256=_sha(b)) for rel, b in files.items()])
    body = json.dumps(data, indent=1).encode("utf-8")
    (pres / "MANIFEST.json").write_bytes(body)
    return body


def _exceptions_a():
    return {str(n): dict(failure="gate %d reads FAIL on the preserved checkpoint" % n,
                         relevance="the failure is shared by the run's start and its end",
                         why_measurable="the run's readings compare physical outcomes at its start and its end",
                         evidence=["row_%d" % n]) for n in range(1, 8)}


def _edit_rows(reg: Path, row_id, **changes):
    data = json.loads(reg.read_text(encoding="utf-8"))
    for r in data["rows"]:
        if r["id"] == row_id:
            r.update(changes)
            for k in [k for k, v in changes.items() if v is None]:
                r.pop(k)
    reg.write_text(json.dumps(data), encoding="utf-8")


def _world(tmp_path, monkeypatch, used=True):
    """The stand-in adoption, its stopped run, its preservation, receipts and rows, and the recovery's inputs (the module
    text). With used=False the adoption is written and never launched."""
    src, root, reg = _src(tmp_path), tmp_path / "pre", _registry(tmp_path)
    p1 = tmp_path / "preserved" / "start"
    _write_files(p1, PRESERVED_A)
    run1, ck1 = p1 / "run", p1 / CK1_REL.rstrip("/")
    (run1 / "args.json").write_text(json.dumps(dict(max_steps=1000, resume=None)), encoding="utf-8")
    (run1 / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=1000, finished_at=time.time() - 1000)),
                                          encoding="utf-8")
    _stats(run1, ((900, 167.0), (1000, 440.0)))
    m1_body = _manifest(p1, PRESERVED_A, step=1000, model_sha256=_sha(PRESERVED_A[CK1_REL + "model.safetensors"]))
    receipt_a = tmp_path / "outcome_a.json"
    receipt_a.write_text(json.dumps(dict(outcome="A", end=1000)), encoding="utf-8")
    message_a, message_r = tmp_path / "message_a.json", tmp_path / "message_r.json"
    message_a.write_text(json.dumps(dict(id=DECISION_A, text=MESSAGE_A)), encoding="utf-8")
    message_r.write_text(json.dumps(dict(id=DECISION_R, text=MESSAGE_R)), encoding="utf-8")
    delegation = tmp_path / "delegation.txt"
    delegation.write_text("H84 OPERATOR DELEGATION (a stand-in)\n\n" + "\n\n".join('"%s"' % s for s in SENTENCES)
                          + "\n\nTHE APPLICATION OF THAT DELEGATION\n", encoding="utf-8")
    template_a = tmp_path / "template_a.json"
    template_a.write_text(json.dumps(
        ["python", "-u", "-m", "mimo_bridge.train_mimo", "--intention", "on", "--report-every", "100",
         "--proprio-token-skip", "1,3", "--latent-body-hierarchy", "shadow", "--caregiver-record-dir", "E:/store",
         "--checkpoint-every-min", "60.0", "--minutes", "0", "--resume", "E:/old/run/checkpoints/ckpt_step_500",
         "--out", "E:/old/run", "--checkpoint-storage", "E:/old/run/checkpoints", "--latent-body-resume",
         "E:/old/latent/latest.pt", "--latent-body-output", "E:/old/latent/output", "--max-steps", "1000"], indent=1),
        encoding="utf-8")
    a_run, a_lat = tmp_path / "experiments" / "adoption_run", tmp_path / "lat_out_a"
    cmd_a = TP.apply_sets(json.loads(template_a.read_text(encoding="utf-8")), [
        ("--resume", str(ck1)), ("--out", str(a_run)), ("--checkpoint-storage", str(a_run / "checkpoints")),
        ("--latent-body-resume", str(p1 / "latent_body" / "latest.pt")), ("--latent-body-output", str(a_lat)),
        ("--max-steps", str(END))])
    fp = Path(TP.new_record(cmd_a, "the new configuration (a stand-in)", "tester", root=root, src=src)).stem
    for n in range(1, 8):
        TP.set_gate(fp, n, "FAIL", "fails on the preserved checkpoint", ["row_%d" % n], "tester", root=root)
    time.sleep(0.01)
    a_route = dict(fingerprint=fp, checkpoint=str(ck1), manifest=str(p1 / "MANIFEST.json"),
                   manifest_sha256=_sha(m1_body), preserved_root=str(tmp_path / "preserved"),
                   model_sha256=_sha(PRESERVED_A[CK1_REL + "model.safetensors"]),
                   latent_sha256=_sha(PRESERVED_A["latent_body/latest.pt"]), predecessor_run=str(run1),
                   outcome_receipt=str(receipt_a), outcome_receipt_sha256=_sha(receipt_a.read_bytes()),
                   decision=DECISION_A, decision_message=str(message_a),
                   decision_text_sha256=_sha(MESSAGE_A.encode("utf-8")), decision_author=AUTHOR,
                   delegation_record=str(delegation), delegation_words=SENTENCES, from_commit=FROM_COMMIT,
                   command_template=str(template_a), command_template_sha256=_sha(template_a.read_bytes()),
                   run_row="adopt_stop_row", start=START_A, end=END, wall_cap_s=WALL_A)
    monkeypatch.setattr(TP, "ADOPTION_ROUTES", (a_route,))
    a_block = dict(
        purpose="one bounded run of the new configuration from the preserved checkpoint, 1,000 to 2,000",
        predecessor=dict(checkpoint=str(ck1), manifest_path=str(p1 / "MANIFEST.json"), manifest_sha256=_sha(m1_body),
                         model_sha256=a_route["model_sha256"], run=str(run1), outcome_receipt=str(receipt_a),
                         outcome_receipt_sha256=a_route["outcome_receipt_sha256"]),
        code_transition=dict(from_commit=FROM_COMMIT, to_commit=TO_COMMIT, commits=[TO_COMMIT]),
        decision=dict(id=DECISION_A, message_sha256=a_route["decision_text_sha256"], author=AUTHOR),
        authorization=AUTH_A, authorized_by=AUTHOR, authorized_at="Oct 13, 2026 5:10 AM PDT",
        start_checkpoint=str(ck1), rollback_checkpoint=str(ck1), start_step=START_A, end_step=END,
        max_model_steps=END - START_A, wall_cap_s=WALL_A,
        run_flags={"--latent-body-resume": str(p1 / "latent_body" / "latest.pt"), "--latent-body-output": str(a_lat)},
        stop_lines="S1", stop_lines_row="adopt_stop_row", stop_criteria=STOP_CRITERIA, completion_jobs_row="adopt_done_row",
        reachability="from 1,000 the first full stats record comes at 1,100; the stop line is first judged before 2,000",
        no_auto_continue="the owner starts the trainer with --max-steps 2000 and no supervisor",
        exceptions=_exceptions_a())
    TP.set_adoption(fp, a_block, "tester", root=root, registry=reg, src=src)
    command_a = tmp_path / "adoption_command.json"            # the adoption's launch command: the recovery's template
    command_a.write_text(json.dumps(cmd_a, indent=1), encoding="utf-8")
    ck_a = a_run / CK2_REL.replace("run/", "", 1).rstrip("/")
    if used:
        a_run.mkdir(parents=True)
        TP.note_attempt(fp, str(a_run), "launcher", root=root, block=TP.ADOPTION, src=src, registry=reg)
        _edit_rows(reg, "adopt_stop_row", status="running")
        TP.mark_launched(fp, str(a_run), "launcher", root=root, block=TP.ADOPTION, src=src, registry=reg)
        # the trainer: trained 1,000 -> 1,200 and was stopped; its end checkpoint and latent body state
        (a_run / "args.json").write_text(json.dumps(dict(resume=str(ck1), max_steps=END, report_every=100)),
                                         encoding="utf-8")
        _stats(a_run, ((1100, 300.0), (1200, 600.0)))
        (a_run / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=STOP, finished_at=time.time() - 50)),
                                               encoding="utf-8")
        _write_files(a_run, {k.replace("run/", "", 1): v for k, v in PRESERVED_R.items() if k.startswith(CK2_REL)})
        a_lat.mkdir()
        (a_lat / "latest.pt").write_bytes(PRESERVED_R["latent_body/latest.pt"])
        _edit_rows(reg, "adopt_stop_row", status="failed", last_run=dict(state="STOPPED"))
        _edit_rows(reg, "adopt_done_row", status="failed", last_run=dict(state="INVALID"))
    # the preservation of the stopped run's end
    p2 = tmp_path / "preserved" / "stop"
    _write_files(p2, PRESERVED_R)
    run2, ck2 = p2 / "run", p2 / CK2_REL.rstrip("/")
    (run2 / "args.json").write_text(json.dumps(dict(resume=str(ck1), max_steps=END, report_every=100)),
                                    encoding="utf-8")
    (run2 / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=STOP, finished_at=time.time() - 50)),
                                          encoding="utf-8")
    _stats(run2, ((1100, 300.0), (1200, 600.0)))
    companion = _sha(PRESERVED_R[COMPANION_REL])
    m2_body = _manifest(p2, PRESERVED_R, run_files=("args.json", "stage_done.json"), checkpoint=str(ck_a), step=STOP,
                        model_sha256=_sha(PRESERVED_R[CK2_REL + "model.safetensors"]), companion_sha256=companion,
                        companion_sha256_in_episode_state=companion, companion_equal=True)
    stopped = tmp_path / "arm_end.json"
    stopped.write_text(json.dumps(dict(state="stopped", step=STOP, run=str(a_run), trainer_exit_code=0,
                                       preserved=dict(verdict="pass", step=STOP), preserved_checkpoint=str(ck2))),
                       encoding="utf-8")
    outcome_r = tmp_path / "outcome_r.json"
    outcome_r.write_text(json.dumps(dict(outcome="[a shorter run] INCOMPLETE - the run stopped at model step 1200")),
                         encoding="utf-8")
    owner_log = tmp_path / "owner.jsonl"
    owner_log.write_text("".join(json.dumps(e) + "\n" for e in (
        dict(event="trainer started", arm="adoption", pid=1, created=T_CREATED, at_s=T_CREATED + 0.2),
        dict(event="trainer ended", arm="adoption", exit_code=0, at_s=T_ENDED))), encoding="utf-8")
    r_run, r_lat = tmp_path / "new_run_r", tmp_path / "lat_out_r"
    cmd_r = TP.apply_sets(cmd_a, [("--resume", str(ck2)), ("--out", str(r_run)),
                                  ("--checkpoint-storage", str(r_run / "checkpoints")),
                                  ("--latent-body-resume", str(p2 / "latent_body" / "latest.pt")),
                                  ("--latent-body-output", str(r_lat))])
    rec = TP._load(TP.record_path(fp, root))
    return dict(src=src, root=root, reg=reg, fp=fp, tmp=tmp_path, ck1=ck1, p1=p1, a_run=a_run, a_lat=a_lat, ck_a=ck_a,
                cmd_a=cmd_a, command_a=command_a, p2=p2, run2=run2, ck2=ck2, manifest2=p2 / "MANIFEST.json",
                manifest2_sha=_sha(m2_body), model2_sha=_sha(PRESERVED_R[CK2_REL + "model.safetensors"]),
                companion_sha=companion, latent2=p2 / "latent_body" / "latest.pt",
                latent2_sha=_sha(PRESERVED_R["latent_body/latest.pt"]), stopped=stopped, outcome_r=outcome_r,
                owner_log=owner_log, message_r=message_r, delegation=delegation, r_run=r_run, r_lat=r_lat,
                cmd_r=cmd_r, code_hash=rec["code_hash"], record_sha=_sha(TP.record_path(fp, root).read_bytes()),
                adoption_sha=_sha(TP.adoption_path(fp, root).read_bytes()))


def _route(w, **changes):
    route = dict(fingerprint=w["fp"], adoption_sha256=w["adoption_sha"], adoption_decision=DECISION_A,
                 adoption_run=str(w["a_run"]), adoption_row="adopt_stop_row", adoption_start=START_A,
                 original_wall_cap_s=WALL_A, record_sha256=w["record_sha"], checkpoint=str(w["ck2"]),
                 manifest=str(w["manifest2"]), manifest_sha256=w["manifest2_sha"],
                 preserved_root=str(w["tmp"] / "preserved"), predecessor_run=str(w["run2"]),
                 model_sha256=w["model2_sha"], companion_sha256=w["companion_sha"], latent_sha256=w["latent2_sha"],
                 stopped_receipt=str(w["stopped"]), stopped_receipt_sha256=_sha(w["stopped"].read_bytes()),
                 outcome_receipt=str(w["outcome_r"]), outcome_receipt_sha256=_sha(w["outcome_r"].read_bytes()),
                 decision=DECISION_R, decision_message=str(w["message_r"]),
                 decision_text_sha256=_sha(MESSAGE_R.encode("utf-8")), decision_author=AUTHOR,
                 delegation_record=str(w["delegation"]), delegation_words=SENTENCES, code_commit=TO_COMMIT,
                 command_template=str(w["command_a"]), command_template_sha256=_sha(w["command_a"].read_bytes()),
                 run_row="recover_stop_row", start=STOP, end=END, wall_cap_s=WALL_R, owner_log=str(w["owner_log"]),
                 trainer_created_s=T_CREATED, trainer_exit_observed_s=T_ENDED)
    route.update(changes)
    return route


def _exceptions_r(w):
    carried = _adoption_block(w)["exceptions"]
    return {n: dict(x, carried_from="adoptions/%s.json gate %s" % (w["fp"], n),
                    applicability="the stopped run trained the same configuration from the adoption's start; gate %s "
                                  "was not measured again, so its failure applies to the preserved end" % n)
            for n, x in carried.items()}


def _block(w, **more):
    b = dict(purpose="the stopped adoption's remainder, 1,200 to 2,000",
             predecessor=dict(adoption=str(TP.adoption_path(w["fp"], w["root"])), adoption_sha256=w["adoption_sha"],
                              adoption_decision=DECISION_A, adoption_run=str(w["a_run"]), stopped_step=STOP,
                              stopped_receipt=str(w["stopped"]), stopped_receipt_sha256=_sha(w["stopped"].read_bytes()),
                              checkpoint=str(w["ck2"]), manifest_path=str(w["manifest2"]),
                              manifest_sha256=w["manifest2_sha"], model_sha256=w["model2_sha"],
                              companion_sha256=w["companion_sha"], run=str(w["run2"]),
                              outcome_receipt=str(w["outcome_r"]),
                              outcome_receipt_sha256=_sha(w["outcome_r"].read_bytes())),
             code=dict(commit=TO_COMMIT[:9], code_hash=w["code_hash"]),
             decision=dict(id=DECISION_R, message_sha256=_sha(MESSAGE_R.encode("utf-8")), author=AUTHOR),
             authorization=AUTH_R, authorized_by=AUTHOR, authorized_at="Oct 13, 2026 11:18 AM PDT",
             start_checkpoint=str(w["ck2"]), rollback_checkpoint=str(w["ck2"]), start_step=STOP, end_step=END,
             max_model_steps=END - STOP, wall_cap_s=WALL_R,
             run_flags={"--latent-body-resume": str(w["latent2"]), "--latent-body-output": str(w["r_lat"])},
             stop_lines="S1, carried", stop_lines_row="recover_stop_row", stop_criteria=STOP_CRITERIA,
             completion_jobs_row="recover_done_row",
             reachability="from 1,200 the first full stats record comes at 1,300; the stop line is first judged up to "
                          "235 model steps in, before the end at 2,000",
             no_auto_continue="the owner starts the trainer with --max-steps 2000 and no supervisor",
             exceptions=_exceptions_r(w))
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
def world(tmp_path, monkeypatch):
    """world(name, used=True, **route_changes): a stand-in world in its own folder, RECOVERY_ROUTES naming its recovery
    with `route_changes` applied (pytest's monkeypatch restores both routes after the test)."""
    def _make(name="w", used=True, **changes):
        w = _world(tmp_path / name, monkeypatch, used=used)
        w["monkeypatch"] = monkeypatch
        monkeypatch.setattr(TP, "RECOVERY_ROUTES", (_route(w, **changes),))
        return w
    return _make


def _pin(w, **changes):
    w["monkeypatch"].setattr(TP, "RECOVERY_ROUTES", (_route(w, **changes),))


def _adoption_block(w):
    return json.loads(TP.adoption_path(w["fp"], w["root"]).read_text(encoding="utf-8"))["block"]


def _write(w, **more):
    return TP.set_recovery(w["fp"], _block(w, **more), "tester", root=w["root"], registry=w["reg"], src=w["src"])


def _refusal(w, cmd=None):
    return TP.launch_refusal(cmd or w["cmd_r"], root=w["root"], src=w["src"], registry=w["reg"])


def _plan(w, cmd=None):
    return TP.launch_plan(cmd or w["cmd_r"], root=w["root"], src=w["src"])


def _folder(w, name):
    run = w["tmp"] / "experiments" / name
    run.mkdir(parents=True, exist_ok=True)
    return run


def _note(w, run, src=True):
    return TP.note_attempt(w["fp"], str(run), "launcher", root=w["root"], block=TP.RECOVERY,
                           src=w["src"] if src else None, registry=w["reg"])


def _mark(w, run, src=True):
    return TP.mark_launched(w["fp"], str(run), "launcher", root=w["root"], block=TP.RECOVERY,
                            src=w["src"] if src else None, registry=w["reg"])


def _held(w):
    return json.loads(TP.recovery_path(w["fp"], w["root"]).read_text(encoding="utf-8"))


def _edit_recovery(w, **changes):
    held = _held(w)
    held["block"].update(changes)
    TP._save(TP.recovery_path(w["fp"], w["root"]), held)


def _rows(w):
    return {r["id"]: r for r in json.loads(w["reg"].read_text(encoding="utf-8"))["rows"]}


def _refused_written(w, needle, **more):
    """Writing the recovery (with `more` in its block) is refused with `needle`, and no recovery file is written."""
    with pytest.raises(TP.PreflightError) as exc:
        _write(w, **more)
    assert needle in str(exc.value), str(exc.value)
    assert not TP.recovery_path(w["fp"], w["root"]).exists()


def _refused_noted(w, needle):
    """Noting an attempt and marking a launch of the recovery are each refused with `needle`; nothing is recorded."""
    run = _folder(w, "run_noted")
    for fn in (_note, _mark):
        with pytest.raises(TP.PreflightError) as exc:
            fn(w, run)
        assert needle in str(exc.value), (fn.__name__, str(exc.value))
    assert not _held(w)["block"].get("attempts") and not _held(w)["block"].get("launched_at")
    assert not any(TP.RECOVERY_ROW_FIELD in r for r in _rows(w).values())


def _refused_at_launch_and_noted(w, needle, cmd=None):
    why = _refusal(w, cmd)
    assert why and needle in why, why
    _refused_noted(w, needle)


def _refused_everywhere(world, needle, damage, name="x", **route_changes):
    """`damage` (applied to a fresh world before the recovery is written) refuses writing it; on a second world, applied
    after a clean write, it refuses the launch and the noting."""
    w = world(name + "_written")
    damage(w)
    if route_changes:
        _pin(w, **route_changes)
    _refused_written(w, needle)
    w2 = world(name + "_launch")
    _write(w2)
    damage(w2)
    if route_changes:
        _pin(w2, **route_changes)
    _refused_at_launch_and_noted(w2, needle)


LISTED = "it launches only as its adoption"
ATTEMPT = "the training row recover_stop_row records this recovery's attempt"


def _tree(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): _sha(p.read_bytes()) for p in sorted(folder.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------ (1) THE ADMITTED CASE
def test_the_recovery_admits_exactly_its_launch_and_keeps_the_adoptions_history(world):
    w = world()
    rec_path, ad_path = TP.record_path(w["fp"], w["root"]), TP.adoption_path(w["fp"], w["root"])
    before = (rec_path.read_bytes(), ad_path.read_bytes(), {k: v for k, v in _rows(w).items()
                                                           if k.startswith("adopt_")})
    assert TP.launch_plan(w["cmd_r"], root=w["root"], src=w["src"])["kind"] == "normal"   # no recovery yet
    b = _write(w)
    held = _held(w)
    assert held["fingerprint"] == w["fp"] and held["record_sha256"] == w["record_sha"]
    assert held["adoption_sha256"] == w["adoption_sha"] and held["route"]["checkpoint"] == str(w["ck2"])
    assert b["role"] == "recovery" and b["written_at_s"] > 0 and b["wall_cap_s"] == WALL_R == 1899
    assert _refusal(w) is None
    plan = _plan(w)
    assert (plan["kind"], plan["block"], plan["fingerprint"], plan["start_step"], plan["max_steps"],
            plan["wall_cap_s"]) == ("continuation", "recovery", w["fp"], STOP, END, WALL_R)
    assert plan["stop_criteria"] == STOP_CRITERIA
    run = _folder(w, "run_r")
    _note(w, run)
    _edit_rows(w["reg"], "recover_stop_row", status="running")     # the run owner's job_started, between note and mark
    _mark(w, run)
    rows = _rows(w)
    kept = rows["recover_stop_row"][TP.RECOVERY_ROW_FIELD]
    assert kept["fingerprint"] == w["fp"] and kept["decision"] == DECISION_R and kept["launched_at"]
    assert [a["run"] for a in kept["attempts"]] == [str(run.resolve())]
    assert TP.ADOPTION_ROW_FIELD not in rows["recover_stop_row"]
    assert rows["recover_done_row"] == dict(id="recover_done_row", status="queued")     # only read
    # THE HISTORY: the record, the used adoption's file and the adoption's rows byte for byte as they were
    assert (rec_path.read_bytes(), ad_path.read_bytes(), {k: v for k, v in rows.items()
                                                         if k.startswith("adopt_")}) == before
    rec = TP._load(rec_path)
    assert [g["status"] for g in rec["gates"]] == ["FAIL"] * 7 and [s["checkpoint"] for s in rec["states"]] == [
        str(w["ck1"])]
    assert all(x["failure"] == _adoption_block(w)["exceptions"][n]["failure"] for n, x in b["exceptions"].items())


def test_the_recovery_is_used_once(world):
    w = world()
    _write(w)
    run = _folder(w, "run_x")
    _note(w, run)
    why = _refusal(w)
    assert why and ATTEMPT in why, why                                  # noted: no second launch, trained or not
    _stats(run, ((1300, 108.0),))
    why = _refusal(w)
    assert why and LISTED in why and "and as its linked recovery (RECOVERY_ROUTES)" in why, why
    assert _plan(w)["kind"] == "normal"
    with pytest.raises(TP.PreflightError, match="the recovery of configuration .* was used"):
        _write(w)
    _mark(w, run)
    with pytest.raises(TP.PreflightError, match="launched already"):
        _mark(w, run)
    with pytest.raises(TP.PreflightError, match="launched already"):
        _note(w, _folder(w, "run_y"))
    # the adoption cannot be written again either: it stays used, and no fresh adoption is admitted
    with pytest.raises(TP.PreflightError, match="the adoption of configuration .* was used"):
        TP.set_adoption(w["fp"], _adoption_block(w), "tester", root=w["root"], registry=w["reg"], src=w["src"])


def test_a_hand_edited_recovery_file_is_refused_by_its_training_row(world):
    w = world()
    _write(w)
    run = _folder(w, "run_first")
    _note(w, run)
    _mark(w, run)
    _stats(run, ((1300, 108.0),))
    new_out = str(w["r_lat"]) + "_second"
    _edit_recovery(w, launched_at=None, launched_at_s=None, run=None, attempts=[],
                   run_flags={"--latent-body-resume": str(w["latent2"]), "--latent-body-output": new_out})
    why = _refusal(w, TP.apply_sets(w["cmd_r"], [("--latent-body-output", new_out)]))
    assert why and "the training row recover_stop_row records this recovery's launch marked" in why, why
    for fn in (_note, _mark):
        with pytest.raises(TP.PreflightError, match="records this recovery's launch marked"):
            fn(w, _folder(w, "run_second"))


def test_the_recovery_file_names_its_own_configuration(world):
    w = world()
    _write(w)
    held = _held(w)
    held["fingerprint"] = OTHER_FP
    TP._save(TP.recovery_path(w["fp"], w["root"]), held)
    _refused_at_launch_and_noted(w, "the recovery file names configuration %s, not %s" % (OTHER_FP, w["fp"]))


def test_a_recovery_file_written_against_another_record_is_refused(world):
    w = world()
    _write(w)
    held = _held(w)
    held["record_sha256"] = "0" * 64
    TP._save(TP.recovery_path(w["fp"], w["root"]), held)
    _refused_at_launch_and_noted(w, "the recovery file was written against the record sha256 %s; the recovery route "
                                    "pins %s" % ("0" * 64, w["record_sha"]))


def test_without_the_recovery_the_configuration_is_never_trained_from_the_stop(world):
    w = world()
    why = _refusal(w)
    assert why and LISTED in why and "and as its linked recovery (RECOVERY_ROUTES), once, from %s to model step %d" % (
        w["ck2"], END) in why, why
    w["monkeypatch"].setattr(TP, "ADOPTION_ROUTES", ())                 # a route listing the recovery alone
    why = _refusal(w)
    assert why and "is the linked recovery's (RECOVERY_ROUTES): it launches only as its recovery" in why, why


def test_the_route_names_the_decisions_one_recovery():
    """The production pins: the values the decision names, compared with this module's own constants; the wall cap is
    the receipts' arithmetic (floor(20683 - (1791913930.4329011 - 1791913178.843792)) = 19931); no file is read."""
    assert len(TP.RECOVERY_ROUTES) == 1
    route = dict(TP.RECOVERY_ROUTES[0])
    assert route == PRODUCTION_ROUTE
    assert route["end"] - route["start"] == 4400
    assert int(math.floor(route["original_wall_cap_s"] - (route["trainer_exit_observed_s"]
                                                          - route["trainer_created_s"]))) == route["wall_cap_s"] == 19931
    assert route["fingerprint"] == TP.ADOPTION_ROUTES[0]["fingerprint"] and route["adoption_decision"] == \
        TP.ADOPTION_ROUTES[0]["decision"] and route["adoption_start"] == TP.ADOPTION_ROUTES[0]["start"] and \
        route["end"] == TP.ADOPTION_ROUTES[0]["end"] and route["original_wall_cap_s"] == TP.ADOPTION_ROUTES[0]["wall_cap_s"]


# ------------------------------------------------------------------------------ (2) WRONG PARENT
def test_a_used_adoption_file_whose_sha256_differs_is_refused(world):
    def damage(w):
        held = json.loads(TP.adoption_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
        held["block"]["note"] = "edited after its use"
        TP._save(TP.adoption_path(w["fp"], w["root"]), held)
    _refused_everywhere(world, "the used adoption's file", damage)


def test_an_adoption_that_was_never_used_is_refused(world):
    """A recovery continues a used adoption's stopped run; an unused adoption launches as itself (here its file's
    sha256 is pinned as it reads, so only the use is missing)."""
    w = world(used=False)
    _pin(w, adoption_sha256=_sha(TP.adoption_path(w["fp"], w["root"]).read_bytes()))
    with pytest.raises(TP.PreflightError) as exc:
        TP.set_recovery(w["fp"], _block(w, predecessor=_pred(w, adoption_sha256=_sha(
            TP.adoption_path(w["fp"], w["root"]).read_bytes()))), "tester", root=w["root"], registry=w["reg"],
            src=w["src"])
    assert "was not used (no marked launch, no attempt that trained past its start)" in str(exc.value)


def test_an_adoption_launched_for_another_run_is_refused(world):
    w = world()
    other = str(w["tmp"] / "experiments" / "another_run")
    _pin(w, adoption_run=other)
    _refused_written(w, "the used adoption's marked launch names run %s; the recovery route pins the run %s" % (
        str(w["a_run"].resolve()), other), predecessor=_pred(w, adoption_run=other))


def test_a_block_naming_another_adoption_file_is_refused(world):
    w = world()
    other = str(w["tmp"] / "pre" / "adoptions" / ("%s.json" % OTHER_FP))
    _refused_written(w, "the recovery's predecessor names the adoption file %s" % other,
                     predecessor=_pred(w, adoption=other))
    w2 = world("launch")
    _write(w2)
    _edit_recovery(w2, predecessor=_pred(w2, adoption=other))
    _refused_at_launch_and_noted(w2, "the recovery's predecessor names the adoption file %s" % other)


def test_the_adoptions_decision_is_pinned(world):
    w = world(adoption_decision="another-adoption")
    _refused_written(w, "continues the adoption of decision another-adoption; the recovery names %s" % DECISION_A)
    _refused_written(w, "the used adoption names decision %s; the recovery route continues the adoption of decision "
                        "another-adoption" % DECISION_A, predecessor=_pred(w, adoption_decision="another-adoption"))


def test_a_record_judged_again_since_the_adoption_is_refused(world):
    def damage(w):
        time.sleep(0.01)
        TP.set_gate(w["fp"], 4, "FAIL", "judged again", ["row_4b"], "tester", root=w["root"])
    _refused_everywhere(world, "(the recovery route's pin): a gate judged again or a state added since then is not "
                               "carried by the recovery", damage)


def test_the_adoption_was_written_against_the_pinned_record(world):
    w = world(record_sha256="7" * 64)
    _refused_written(w, "the used adoption was written against the record sha256 %s; the recovery route pins %s" % (
        w["record_sha"], "7" * 64))


def test_a_configuration_the_route_does_not_name_is_refused(world):
    w = world(fingerprint=OTHER_FP)
    _refused_written(w, "configuration %s is not one" % w["fp"])
    _pin(w)
    _write(w)
    _pin(w, fingerprint=OTHER_FP)
    _refused_noted(w, "configuration %s is not one" % w["fp"])


def test_a_start_whose_lineage_does_not_reach_the_adoptions_start_is_refused(world):
    """The preserved run's args.json resumes another checkpoint (its MANIFEST recording that file): the recovery's start
    is not the same lineage."""
    w = world()
    (w["run2"] / "args.json").write_text(json.dumps(dict(resume=str(w["tmp"] / "else" / "run" / "checkpoints" /
                                                                   "ckpt_step_1000"), max_steps=END)), encoding="utf-8")
    body = _manifest(w["p2"], PRESERVED_R, run_files=("args.json", "stage_done.json"), checkpoint=str(w["ck_a"]),
                     step=STOP, model_sha256=w["model2_sha"], companion_sha256=w["companion_sha"],
                     companion_sha256_in_episode_state=w["companion_sha"], companion_equal=True)
    _pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "does not continue the model state %s the used adoption started from" % w["ck1"],
                     predecessor=_pred(w, manifest_sha256=_sha(body)))


@pytest.mark.parametrize("row, changes, needle", [
    ("adopt_stop_row", dict(status="queued"), "the used adoption's training row adopt_stop_row reads status queued"),
    ("adopt_stop_row", dict(last_run=dict(state="PASS")), "and its runner's state PASS"),
    ("adopt_stop_row", {TP.ADOPTION_ROW_FIELD: None}, "does not record its launch of run"),
    ("adopt_done_row", dict(status="done"), "the used adoption's completion row adopt_done_row reads status done"),
], ids=["training-requeued", "training-pass", "training-record-removed", "completion-done"])
def test_the_used_adoptions_rows_keep_their_ends(world, row, changes, needle):
    _refused_everywhere(world, needle, lambda w: _edit_rows(w["reg"], row, **changes))


def test_the_used_adoptions_rows_must_exist(world):
    w = world(adoption_row="no_such_row")
    _refused_written(w, "the used adoption's training row no_such_row is not in the test registry")


# ------------------------------------------------------------------------------ (3) WRONG CHECKPOINT
def test_the_original_end_checkpoint_instead_of_its_preserved_copy_is_refused(world):
    w = world()
    _refused_written(w, "admits only the stopped run's preserved end checkpoint", start_checkpoint=str(w["ck_a"]),
                     rollback_checkpoint=str(w["ck_a"]), predecessor=_pred(w, checkpoint=str(w["ck_a"])))


def test_the_adoptions_own_start_is_refused(world):
    w = world()
    _refused_written(w, "admits only the stopped run's preserved end checkpoint", start_checkpoint=str(w["ck1"]),
                     rollback_checkpoint=str(w["ck1"]), predecessor=_pred(w, checkpoint=str(w["ck1"])),
                     start_step=START_A, max_model_steps=END - START_A)


def test_a_launch_from_another_checkpoint_is_not_the_recovery(world):
    w = world()
    _write(w)
    cmd = TP.apply_sets(w["cmd_r"], [("--resume", str(w["ck_a"]))])
    why = _refusal(w, cmd)
    assert why and LISTED in why, why
    assert _plan(w, cmd)["kind"] == "normal"
    why = TP._recovery_refusal(cmd, TP._load(TP.record_path(w["fp"], w["root"])), _held(w),
                               TP.configuration(cmd, w["src"]), w["root"], w["reg"])
    assert why and "not the recovery's start" in why, why


PIN_CHANGES = [
    ("checkpoint", lambda w: str(w["ck_a"]), "admits only the stopped run's preserved end checkpoint"),
    ("manifest", lambda w: str(w["tmp"] / "elsewhere" / "MANIFEST.json"), "pins the preservation MANIFEST"),
    ("predecessor_run", lambda w: str(w["tmp"] / "elsewhere" / "run"), "pins the preserved run"),
    ("adoption_run", lambda w: str(w["tmp"] / "elsewhere" / "run"), "pins the used adoption's run"),
    ("stopped_receipt", lambda w: str(w["tmp"] / "elsewhere" / "arm_end.json"), "pins the stopped receipt"),
    ("outcome_receipt", lambda w: str(w["tmp"] / "elsewhere" / "outcome.json"), "pins the outcome receipt"),
    ("manifest_sha256", lambda w: "0" * 64, "pins the preservation MANIFEST's sha256 " + "0" * 64),
    ("model_sha256", lambda w: "1" * 64, "pins the model's sha256 " + "1" * 64),
    ("companion_sha256", lambda w: "2" * 64, "pins the worlds companion's sha256 " + "2" * 64),
    ("outcome_receipt_sha256", lambda w: "3" * 64, "pins the outcome receipt's sha256 " + "3" * 64),
    ("stopped_receipt_sha256", lambda w: "4" * 64, "pins the stopped receipt's sha256 " + "4" * 64),
    ("adoption_sha256", lambda w: "5" * 64, "pins the used adoption file's sha256 " + "5" * 64),
    ("latent_sha256", lambda w: "6" * 64, "pins the latent body state's sha256 " + "6" * 64),
    ("decision", lambda w: "another-decision", "pins the decision another-decision; the recovery names " + DECISION_R),
    ("decision_text_sha256", lambda w: "8" * 64, "pins the decision message's text sha256 " + "8" * 64),
    ("decision_author", lambda w: "someone else", "pins the decision's author 'someone else'"),
    ("start", lambda w: 1199, "continues the run stopped at model step 1199"),
    ("end", lambda w: 1900, "ends at model step 1900; the recovery's end_step reads 2000"),
    ("wall_cap_s", lambda w: 1898, "pins the remaining wall cap 1898 s; the recovery's wall_cap_s reads 1899"),
    ("run_row", lambda w: "other_row", "pins the training row other_row; the recovery's stop_lines_row reads "
                                       "'recover_stop_row'"),
    ("code_commit", lambda w: "9" * 40, "runs the trainer code of commit " + "9" * 40),
]


@pytest.mark.parametrize("key, value, needle", PIN_CHANGES, ids=[c[0] for c in PIN_CHANGES])
def test_each_pinned_value_is_compared_exactly(world, key, value, needle):
    w = world("written")
    _pin(w, **{key: value(w)})
    _refused_written(w, needle)
    w2 = world("launch")
    _write(w2)
    _pin(w2, **{key: value(w2)})
    _refused_at_launch_and_noted(w2, needle)


def test_a_predecessor_model_whose_sha256_differs_is_refused(world):
    _refused_everywhere(world, "the predecessor's model", lambda w: (w["ck2"] / "model.safetensors").write_bytes(b"x"))


def test_a_latent_body_state_other_than_the_stopped_runs_is_refused(world):
    """The adoption's start's latent body state (the preserved 1,000), not the stopped run's."""
    w = world()
    other = w["p1"] / "latent_body" / "latest.pt"
    _refused_written(w, "pins the latent body state's sha256 %s; the state the recovery resumes, %s, reads sha256 %s" % (
        w["latent2_sha"], other, _sha(other.read_bytes())),
        run_flags={"--latent-body-resume": str(other), "--latent-body-output": str(w["r_lat"])})


@pytest.mark.parametrize("changes, needle", [
    (dict(companion_sha256="e" * 64), "records the worlds companion %s" % ("e" * 64)),
    (dict(companion_equal=False), "equal: False"),
    (dict(companion_sha256_in_episode_state="e" * 64), "names %s, equal" % ("e" * 64)),
    (dict(checkpoint="E:/another/run/checkpoints/ckpt_step_1200"), "records the source checkpoint "
                                                                   "E:/another/run/checkpoints/ckpt_step_1200"),
    (dict(step=1100), "at model step 1100"),
    (dict(verdict="fail"), "reads verdict fail; a recovery starts only from a preservation that passed"),
], ids=["companion", "companion-equal", "companion-in-episode-state", "source", "step", "verdict"])
def test_the_preservation_names_the_stopped_runs_end_and_its_worlds(world, changes, needle):
    w = world()
    fields = dict(checkpoint=str(w["ck_a"]), step=STOP, model_sha256=w["model2_sha"], companion_sha256=w["companion_sha"],
                  companion_sha256_in_episode_state=w["companion_sha"], companion_equal=True)
    fields.update(changes)
    body = _manifest(w["p2"], PRESERVED_R, run_files=("args.json", "stage_done.json"), **fields)
    _pin(w, manifest_sha256=_sha(body))
    _refused_written(w, needle, predecessor=_pred(w, manifest_sha256=_sha(body)))


def test_a_companion_file_other_than_the_pinned_is_refused(world):
    w = world()
    other = {k: (b"other worlds" if k == COMPANION_REL else v) for k, v in PRESERVED_R.items()}
    _write_files(w["p2"], other)
    body = _manifest(w["p2"], other, run_files=("args.json", "stage_done.json"), checkpoint=str(w["ck_a"]), step=STOP,
                     model_sha256=w["model2_sha"], companion_sha256=w["companion_sha"],
                     companion_sha256_in_episode_state=w["companion_sha"], companion_equal=True)
    _pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "companion files ['%s']" % _sha(b"other worlds"), predecessor=_pred(w, manifest_sha256=_sha(body)))


def test_the_preserved_args_json_is_the_one_the_manifest_records(world):
    def edited(w):
        args = json.loads((w["run2"] / "args.json").read_text(encoding="utf-8"))
        (w["run2"] / "args.json").write_text(json.dumps(dict(args, note="edited")), encoding="utf-8")
    _refused_everywhere(world, "the preserved run's args.json", edited)


def test_a_preserved_run_launched_for_another_end_is_refused(world):
    w = world()
    (w["run2"] / "args.json").write_text(json.dumps(dict(resume=str(w["ck1"]), max_steps=1800)), encoding="utf-8")
    body = _manifest(w["p2"], PRESERVED_R, run_files=("args.json", "stage_done.json"), checkpoint=str(w["ck_a"]),
                     step=STOP, model_sha256=w["model2_sha"], companion_sha256=w["companion_sha"],
                     companion_sha256_in_episode_state=w["companion_sha"], companion_equal=True)
    _pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "and ends at 1800; the used adoption started from", predecessor=_pred(w, manifest_sha256=_sha(
        body)))


@pytest.mark.parametrize("changes, needle", [
    (dict(state="completed", step=END), "reads state completed at model step 2000"),
    (dict(trainer_exit_code=1), "trainer exit code 1"),
    (dict(preserved=dict(verdict="fail")), "preservation fail"),
    (dict(run="E:/another/run"), "(run E:/another/run"),
    (dict(preserved_checkpoint="E:/another/ckpt_step_1200"), "at E:/another/ckpt_step_1200"),
], ids=["completed", "exit-code", "preservation", "run", "preserved-checkpoint"])
def test_the_stopped_receipt_reads_a_stop_at_the_start_with_its_end_preserved(world, changes, needle):
    """A run that reached its end, or whose stop was not cleanly saved and preserved, leaves nothing for a recovery: it is
    never presented as a stopped predecessor (no successful predecessor is made)."""
    w = world()
    data = json.loads(w["stopped"].read_text(encoding="utf-8"))
    data.update(changes)
    w["stopped"].write_text(json.dumps(data), encoding="utf-8")
    sha = _sha(w["stopped"].read_bytes())
    _pin(w, stopped_receipt_sha256=sha)
    _refused_written(w, needle, predecessor=_pred(w, stopped_receipt_sha256=sha))


def test_a_stopped_receipt_whose_sha256_differs_is_refused(world):
    """The receipt's bytes changed after the block names its sha256 (the route's pin): refused by the file's own hash."""
    w = world()
    named = _sha(w["stopped"].read_bytes())
    w["stopped"].write_bytes(w["stopped"].read_bytes() + b"\n")
    _refused_written(w, "the stopped receipt %s reads sha256" % w["stopped"],
                     predecessor=_pred(w, stopped_receipt_sha256=named))
    w2 = world("launch")
    _write(w2)
    w2["stopped"].write_bytes(w2["stopped"].read_bytes() + b"\n")
    _refused_at_launch_and_noted(w2, "the stopped receipt %s reads sha256" % w2["stopped"])


def test_a_manifest_that_changed_after_the_recovery_is_refused(world):
    w = world()
    _write(w)
    w["manifest2"].write_bytes(w["manifest2"].read_bytes() + b"\n")
    _refused_at_launch_and_noted(w, "(the preservation record changed, or it is another)")


def test_an_outcome_that_is_not_INCOMPLETE_is_refused(world):
    w = world()
    w["outcome_r"].write_text(json.dumps(dict(outcome="A. CANDIDATE FOR ADOPTION")), encoding="utf-8")
    sha = _sha(w["outcome_r"].read_bytes())
    _pin(w, outcome_receipt_sha256=sha)
    _refused_written(w, "the used adoption's result is INCOMPLETE and the recovery keeps it",
                     predecessor=_pred(w, outcome_receipt_sha256=sha))


def test_an_outcome_receipt_whose_sha256_differs_is_refused(world):
    """The receipt's bytes changed after the block names its sha256 (the route's pin): refused when written, at launch
    and when noted, by the file's own hash."""
    w = world()
    named = _sha(w["outcome_r"].read_bytes())
    w["outcome_r"].write_bytes(w["outcome_r"].read_bytes() + b"\n")
    _refused_written(w, "the predecessor's outcome receipt %s reads sha256" % w["outcome_r"],
                     predecessor=_pred(w, outcome_receipt_sha256=named))
    w2 = world("launch")
    _write(w2)
    w2["outcome_r"].write_bytes(w2["outcome_r"].read_bytes() + b"\n")
    _refused_at_launch_and_noted(w2, "the predecessor's outcome receipt %s reads sha256" % w2["outcome_r"])


def test_a_preserved_run_that_did_not_end_at_the_stop_is_refused(world):
    def damage(w):
        (w["run2"] / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=END, finished_at=1.0)),
                                                   encoding="utf-8")
    _refused_everywhere(world, "did not end with its own save at model step 1200", damage)


def test_every_preserved_file_is_hashed_when_written_and_at_launch(world):
    w = world()
    (w["ck2"] / "optimizer.pt").write_bytes(b"other bytes")
    _refused_written(w, "the preserved file %s" % (w["p2"] / CK2_REL / "optimizer.pt"))
    w2 = world("launch")
    _write(w2)
    (w2["p2"] / COMPANION_REL).write_bytes(b"other worlds")
    why = _refusal(w2)
    assert why and "the preserved file %s" % (w2["p2"] / COMPANION_REL) in why, why


# ------------------------------------------------------------------------------ (4) CHANGED TRAINING CONFIGURATION
def test_a_command_whose_configuration_differs_is_refused_naming_each_difference(world):
    w = world()
    _write(w)
    flag = TP.apply_sets(w["cmd_r"], [("--proprio-token-skip", "3")])
    why = _refusal(w, flag)
    assert why and "is not the recovery's; a recovery runs exactly its record's code and settings" in why, why
    assert "flag --proprio-token-skip: 1,3 -> 3" in why
    (w["src"] / "lem" / "model" / "core.py").write_text("x = 2\n", encoding="utf-8")
    why = _refusal(w)
    assert why and "is not the recovery's" in why and "code: 1 file(s) differ: lem/model/core.py" in why, why
    # once the recovery is used, the same commands are judged exactly as before (no record of their configuration)
    (w["src"] / "lem" / "model" / "core.py").write_text("x = 1\n", encoding="utf-8")
    _mark(w, _folder(w, "run_used"))
    why = _refusal(w, flag)
    assert why and "has no preflight record" in why and "is not the recovery's" not in why, why


def test_another_launch_from_another_checkpoint_is_judged_as_before(world):
    w = world()
    _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd_r"], [("--proprio-token-skip", "3"), ("--resume", str(w["ck_a"]))]))
    assert why and "has no preflight record" in why and "is not the recovery's" not in why, why


def test_a_changed_checkout_code_is_refused_when_written_and_when_noted(world):
    w = world()
    (w["src"] / "lem" / "model" / "core.py").write_text("x = 2\n", encoding="utf-8")
    _refused_written(w, "the checkout's trainer code")
    w2 = world("noted")
    _write(w2)
    (w2["src"] / "lem" / "model" / "core.py").write_text("x = 2\n", encoding="utf-8")
    _refused_noted(w2, "the checkout's trainer code")


@pytest.mark.parametrize("sets", [[("--caregiver-record-dir", "E:/another_store")], [("--report-every", "50")],
                                  [("--minutes", "5")]], ids=["caregiver-record-dir", "report-every", "minutes"])
def test_every_flag_but_the_replaced_ones_reads_as_the_adoptions_launch(world, sets):
    w = world()
    _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd_r"], sets))
    assert why and "the launch's %s reads %s; the command template reads" % sets[0] in why, why


def test_the_stop_criteria_are_the_adoptions(world):
    w = world()
    changed = [dict(STOP_CRITERIA[0], line=0.05)]
    _refused_written(w, "the recovery's stop criteria are not the used adoption's", stop_criteria=changed)
    w2 = world("launch")
    _write(w2)
    _edit_recovery(w2, stop_criteria=changed)
    _refused_at_launch_and_noted(w2, "the recovery's stop criteria are not the used adoption's")


def test_the_command_template_reads_its_pinned_sha256(world):
    w = world()
    w["command_a"].write_text(w["command_a"].read_text(encoding="utf-8") + " ", encoding="utf-8")
    _refused_written(w, "the command template %s reads sha256" % w["command_a"])


def test_the_blocks_code_hash_is_the_records(world):
    w = world()
    _refused_written(w, "the recovery's code names code hash %s" % ("c" * 64),
                     code=dict(commit=TO_COMMIT, code_hash="c" * 64))


def test_a_record_whose_settings_were_edited_is_refused(world):
    w = world()
    rec = TP._load(TP.record_path(w["fp"], w["root"]))
    rec["code_hash"] = "0" * 64
    TP._save(TP.record_path(w["fp"], w["root"]), rec)
    _refused_written(w, "its settings or code were changed after it was written")


# ------------------------------------------------------------------------------ (5) REUSED AUTHORIZATION
def test_the_adoptions_decision_is_not_the_recoverys(world):
    w = world(decision=DECISION_A)
    with pytest.raises(TP.PreflightError) as exc:
        _write(w, decision=dict(id=DECISION_A, message_sha256=_sha(MESSAGE_R.encode("utf-8")), author=AUTHOR),
               authorization=AUTH_R.replace(DECISION_R, DECISION_A))
    assert "the decision %s already authorized the adoption" % DECISION_A in str(exc.value), str(exc.value)


def test_the_adoptions_words_are_refused(world):
    _refused_written(world(), "the recovery's authorization and the authorization words of the adoption",
                     authorization=AUTH_A + "\n" + DECISION_R)


def test_the_adoptions_authorized_time_is_refused(world):
    w = world()
    _refused_written(w, "the recovery's authorized_at repeats the used adoption's", authorized_at="Oct 13, 2026  5:10 AM "
                                                                                                "PDT")
    _write(w)
    _edit_recovery(w, authorized_at="Oct 13, 2026 5:10 AM PDT")
    _refused_at_launch_and_noted(w, "the recovery's authorized_at repeats the used adoption's")


def test_a_decision_already_used_by_another_recovery_is_refused(world):
    def damage(w):
        TP._save(TP.recovery_path(OTHER_FP, w["root"]), dict(fingerprint=OTHER_FP, block=dict(
            authorization="other words", decision=dict(id=DECISION_R, message_sha256="f" * 64))))
    _refused_everywhere(world, "the decision %s already authorized the recovery" % DECISION_R, damage)


def test_a_decision_message_already_used_is_refused(world):
    w = world()
    TP._save(TP.recovery_path(OTHER_FP, w["root"]), dict(fingerprint=OTHER_FP, block=dict(
        authorization="other words", decision=dict(id="another", message_sha256=_sha(MESSAGE_R.encode("utf-8"))))))
    _refused_written(w, "the decision message whose sha256 reads %s already authorized" % _sha(
        MESSAGE_R.encode("utf-8")))


def test_authorization_words_that_do_not_name_the_decision_are_refused(world):
    _refused_written(world(), "the recovery's authorization does not name its decision %s" % DECISION_R,
                     authorization=QUOTED)


@pytest.mark.parametrize("damage, needle", [
    (lambda w: w["message_r"].write_text(json.dumps(dict(id=DECISION_R, text=MESSAGE_R + " edited")),
                                         encoding="utf-8"), "holds a text whose sha256 reads"),
    (lambda w: w["delegation"].write_text("nothing", encoding="utf-8"), "does not hold the operator's sentence"),
], ids=["message-text", "delegation"])
def test_the_decision_message_and_the_delegation_record_are_read(world, damage, needle):
    _refused_everywhere(world, needle, damage)


def test_the_authorization_holds_each_delegation_sentence_verbatim(world):
    _refused_written(world(), "the recovery's authorization does not hold the operator's delegation sentence",
                     authorization=AUTH_R.replace(SENTENCES[1], SENTENCES[1].lower()))


# ------------------------------------------------------------------------------ (6) EXCESS STEPS OR TIME
def test_the_launch_ends_exactly_at_the_adoptions_end(world):
    w = world()
    _write(w)
    for steps in (str(END + 1), str(END - 1), str(STOP)):
        why = _refusal(w, TP.apply_sets(w["cmd_r"], [("--max-steps", steps)]))
        assert why and "the launch is not bounded as declared: --max-steps reads %s" % steps in why, why
    bare = list(w["cmd_r"])
    del bare[bare.index("--max-steps"):bare.index("--max-steps") + 2]
    why = _refusal(w, bare)
    assert why and "--max-steps reads absent" in why, why


def test_an_end_beyond_the_adoptions_is_refused(world):
    _refused_written(world(), "ends at model step 2000; the recovery's end_step reads 2001", end_step=END + 1,
                     max_model_steps=END + 1 - STOP)


def test_max_model_steps_other_than_end_minus_start_is_refused(world):
    needle = "max_model_steps 801 is not its end_step minus its start_step (2000 - 1200 = 800)"
    w = world()
    _refused_written(w, needle, max_model_steps=801)
    _write(w)
    _edit_recovery(w, max_model_steps=801)
    _refused_at_launch_and_noted(w, needle)


@pytest.mark.parametrize("cap", [WALL_R + 1, WALL_A, 1.0e9], ids=["one-second-more", "renewed", "huge"])
def test_a_wall_cap_above_the_remaining_time_is_refused(world, cap):
    _refused_written(world(), "pins the remaining wall cap %d s; the recovery's wall_cap_s reads" % WALL_R,
                     wall_cap_s=cap)


@pytest.mark.parametrize("cap", [0, -1.0, "1899", float("nan"), True], ids=["zero", "negative", "text", "nan", "bool"])
def test_the_wall_cap_is_a_number_of_seconds_above_zero(world, cap):
    _refused_written(world(), "wall_cap_s must be a number of seconds above 0", wall_cap_s=cap)


@pytest.mark.parametrize("events, needle", [
    ([dict(event="trainer started", created=T_CREATED)], "records none for its 'trainer ended' event(s)"),
    ([dict(event="trainer started", created=T_CREATED), dict(event="trainer ended", at_s=T_ENDED),
      dict(event="trainer ended", at_s=T_ENDED + 60)], "records [%r, %r] for its 'trainer ended'" % (T_ENDED,
                                                                                                  T_ENDED + 60)),
    ([dict(event="trainer started", created=T_CREATED + 30), dict(event="trainer ended", at_s=T_ENDED)],
     "records [%r] for its 'trainer started' event(s)" % (T_CREATED + 30)),
], ids=["no-end", "two-ends", "another-start"])
def test_the_wall_cap_is_read_from_the_run_owners_receipts(world, events, needle):
    def damage(w):
        w["owner_log"].write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    _refused_everywhere(world, needle, damage)


def test_a_missing_run_owner_log_is_refused(world):
    w = world()
    w["owner_log"].unlink()
    _refused_written(w, "the run owner's log %s cannot be read" % w["owner_log"])


def test_a_route_wall_cap_that_is_not_the_remaining_time_is_refused(world):
    """A route whose pinned cap is not floor(original - elapsed): the arithmetic itself is checked."""
    w = world(wall_cap_s=WALL_R + 1)
    _refused_written(w, "the recovery route's wall cap %d s is not the used adoption's %d s less its trainer's 100.500 "
                        "s" % (WALL_R + 1, WALL_A), wall_cap_s=WALL_R + 1)


def test_the_used_adoptions_original_wall_cap_is_pinned(world):
    w = world(original_wall_cap_s=WALL_A + 1, wall_cap_s=WALL_R + 1)
    _refused_written(w, "the used adoption's wall cap reads %d; the recovery's remaining cap is computed from the "
                        "original %d s" % (WALL_A, WALL_A + 1), wall_cap_s=WALL_R + 1)


def test_a_start_step_other_than_the_stop_is_refused(world):
    _refused_written(world(), "starts at model step 1200; the recovery's start_step reads 1199", start_step=STOP - 1,
                     max_model_steps=END - STOP + 1)


def test_the_adoptions_code_is_the_recoverys(world):
    """The route's commit named by the block, but not the commit the used adoption ran: refused."""
    w = world(code_commit="9" * 40)
    _refused_written(w, "the used adoption's code transition ends at %r; the recovery runs the code of commit %s" % (
        TO_COMMIT, "9" * 40), code=dict(commit="9" * 40, code_hash=w["code_hash"]))


def test_an_adoption_with_another_span_is_refused(world):
    w = world(adoption_start=START_A + 1)
    _refused_written(w, "the used adoption ran from model step %d to %d; the recovery route continues the adoption from "
                        "%d" % (START_A, END, START_A + 1))


# ------------------------------------------------------------------------------ (7) THE GATES
def test_an_inherited_exception_is_carried_unchanged(world):
    w = world()
    for field in ("failure", "relevance", "why_measurable", "evidence"):
        exc = _exceptions_r(w)
        exc["4"] = dict(exc["4"], **{field: ["another"] if field == "evidence" else "rewritten"})
        _refused_written(w, "the recovery's exception for gate 4 Local stability changes the used adoption's %s" % field,
                         exceptions=exc)


@pytest.mark.parametrize("field", TP.RECOVERY_CARRIED_FIELDS)
def test_each_exception_writes_where_it_comes_from_and_its_applicability(world, field):
    w = world()
    exc = _exceptions_r(w)
    exc["2"] = {k: v for k, v in exc["2"].items() if k != field}
    _refused_written(w, "the recovery's exception for gate 2 Gradient flow lacks its %s" % field.replace("_", " "),
                     exceptions=exc)


def test_an_exception_never_reads_a_status_other_than_fail(world):
    w = world()
    exc = _exceptions_r(w)
    exc["5"] = dict(exc["5"], status="PASS")
    _refused_written(w, "the recovery's exception for gate 5 reads status PASS", exceptions=exc)


def test_a_gate_the_used_adoption_has_no_exception_for_is_refused(world):
    """An exception the used adoption never wrote cannot be carried (here the adoption file is edited and its new
    sha256 pinned, so only the missing exception differs)."""
    w = world()
    b = _block(w)
    held = json.loads(TP.adoption_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
    held["block"]["exceptions"].pop("3")
    TP._save(TP.adoption_path(w["fp"], w["root"]), held)
    sha = _sha(TP.adoption_path(w["fp"], w["root"]).read_bytes())
    _pin(w, adoption_sha256=sha)
    b["predecessor"] = dict(b["predecessor"], adoption_sha256=sha)
    with pytest.raises(TP.PreflightError) as exc:
        TP.set_recovery(w["fp"], b, "tester", root=w["root"], registry=w["reg"], src=w["src"])
    assert "gate 3 Scale/conditioning: the used adoption names no exception for it, so none is carried" in str(
        exc.value), str(exc.value)


def test_a_gate_without_its_exception_is_refused(world):
    w = world()
    exc = {n: x for n, x in _exceptions_r(w).items() if n != "6"}
    _refused_written(w, "gate 6 Objective/test alignment reads FAIL and the recovery names no complete exception",
                     exceptions=exc)


def test_a_gate_judged_PASS_is_refused_even_with_the_record_pinned(world):
    """A gate judged PASS after the adoption: the record's sha256 refuses it, and with the route pinned to the edited
    record (a reviewer's mistake) the gate check still refuses it: no inherited failure becomes a new pass."""
    w = world()
    TP.set_gate(w["fp"], 7, "PASS", "claimed without a run", ["any text"], "tester", root=w["root"])
    sha = _sha(TP.record_path(w["fp"], w["root"]).read_bytes())
    _pin(w, record_sha256=sha)
    _refused_written(w, "gate 7 End-to-end causal dry run reads PASS; a recovery launches only with every gate FAIL "
                        "(the used adoption's decision, which the recovery keeps")


def test_the_exceptions_form(world):
    w = world()
    exc = dict(_exceptions_r(w), x=dict(_exceptions_r(w)["1"]))
    _refused_written(w, "the recovery's exception 'x' is not a gate number", exceptions=exc)
    _refused_written(w, "the recovery's exception 9 is not one of the seven gates",
                     exceptions=dict(_exceptions_r(w), **{"9": _exceptions_r(w)["1"]}))
    _refused_written(w, "the recovery's exception for gate 5 must be", exceptions=dict(_exceptions_r(w), **{"5": "x"}))
    _refused_written(w, "the recovery's exceptions must be a block", exceptions=[["1", "x"]])


# ------------------------------------------------------------------------------ (8) THE ROWS, THE LAUNCH, THE FORM
@pytest.mark.parametrize("more, needle", [
    (dict(completion_jobs_row="recover_stop_row"), "the recovery's completion_jobs_row names the training row "
                                                   "recover_stop_row"),
    (dict(completion_jobs_row="adopt_stop_row"), "the recovery's completion_jobs_row names the training row "
                                                 "adopt_stop_row"),
    (dict(completion_jobs_row="adopt_done_row"), "the recovery's completion jobs' row is the used adoption's "
                                                 "adopt_done_row"),
    (dict(completion_jobs_row="missing_row"), "the completion jobs' row missing_row is not in the test registry"),
    (dict(completion_jobs_row="running_row"), "the completion jobs' row running_row reads status running; a "
                                              "recovery's row"),
], ids=["own-training-row", "adoption-training-row", "adoption-completion-row", "missing", "running"])
def test_the_completion_row_is_its_own_and_declared(world, more, needle):
    _refused_written(world(), needle, **more)


def test_the_training_row_must_read_written_or_queued(world):
    w = world()
    _edit_rows(w["reg"], "recover_stop_row", status="done")
    _refused_written(w, "the training row recover_stop_row reads status done; a recovery's row")


def test_a_failed_attempt_refuses_a_second_launch(world):
    w = world()
    _write(w)
    _note(w, _folder(w, "run_failed"))
    assert TP.recovery_available(_held(w)["block"])                     # the recovery file: no use
    retry = TP.apply_sets(w["cmd_r"], [("--latent-body-output", str(w["tmp"] / "lat_retry"))])
    why = _refusal(w, retry)
    assert why and ATTEMPT in why, why
    with pytest.raises(TP.PreflightError, match=ATTEMPT):
        _note(w, _folder(w, "run_retry"))


def test_a_row_recording_another_recovery_is_refused(world):
    w = world()
    _edit_rows(w["reg"], "recover_stop_row", **{TP.RECOVERY_ROW_FIELD: dict(fingerprint=OTHER_FP, attempts=[])})
    _refused_written(w, "the training row recover_stop_row records the launch of another recovery")


@pytest.mark.parametrize("flag, where", [("--out", "a_run"), ("--checkpoint-storage", "a_run"),
                                         ("--latent-body-output", "a_lat")],
                         ids=["out", "checkpoint-storage", "latent-output"])
def test_the_launch_never_writes_into_the_stopped_run(world, flag, where):
    w = world()
    inside = w[where] / "recovery_inside"
    if flag == "--latent-body-output":
        _write(w, run_flags={"--latent-body-resume": str(w["latent2"]), "--latent-body-output": str(inside)})
    else:
        _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd_r"], [(flag, str(inside))]))
    assert why and "the launch's %s %s lies in the used adoption's" % (flag, inside) in why, why


@pytest.mark.parametrize("flag", ["--out", "--checkpoint-storage"])
def test_the_launch_writes_new_folders_outside_the_preservations(world, flag):
    w = world()
    _write(w)
    inside = w["tmp"] / "preserved" / "new_folder"
    why = _refusal(w, TP.apply_sets(w["cmd_r"], [(flag, str(inside))]))
    assert why and "the launch's %s %s lies in the preserved folder" % (flag, inside) in why, why
    why = _refusal(w, TP.apply_sets(w["cmd_r"], [(flag, str(w["run2"]))]))
    assert why and "exists already: a recovery writes a new folder" in why, why


def test_the_run_flags_the_recovery_names_are_carried_exactly(world):
    w = world()
    _write(w)
    for flag, value in (("--latent-body-output", str(w["r_lat"]) + "_other"),
                        ("--latent-body-resume", str(w["latent2"]) + ".other")):
        why = _refusal(w, TP.apply_sets(w["cmd_r"], [(flag, value)]))
        assert why and "the launch's %s reads %s" % (flag, value) in why and "(run_flags, compared exactly)" in why, why


def test_the_run_flags_name_both_latent_body_flags_and_a_new_folder(world):
    w = world()
    _refused_written(w, "the recovery's run_flags name --latent-body-output",
                     run_flags={"--latent-body-resume": str(w["latent2"])})
    w["r_lat"].mkdir()
    _refused_written(w, "exists already; a recovery writes a new folder")


@pytest.mark.parametrize("key", TP.RECOVERY_FIELDS)
def test_every_field_is_required(world, key):
    w = world()
    b = _block(w)
    b.pop(key)
    with pytest.raises(TP.PreflightError, match="the recovery's %s is missing" % key.replace("_", " ")):
        TP.set_recovery(w["fp"], b, "tester", root=w["root"], registry=w["reg"], src=w["src"])


@pytest.mark.parametrize("change, needle", [
    (lambda w: dict(predecessor=_pred(w, stopped_step="1200")), "predecessor stopped_step must be a whole number"),
    (lambda w: dict(predecessor=_pred(w, adoption_sha256="ABC")), "the recovery's predecessor adoption_sha256 must be "
                                                                  "a sha256"),
    (lambda w: dict(predecessor=_pred(w, adoption_run=None)), "the recovery's predecessor adoption_run must be text"),
    (lambda w: dict(code=dict(commit=TO_COMMIT)), "the recovery's code code_hash must be text"),
    (lambda w: dict(code=["a"]), "the recovery's code must be a block"),
    (lambda w: dict(role="adoption"), "the recovery's role reads adoption"),
    (lambda w: dict(repeats=2), "a recovery is used once"),
    (lambda w: dict(target_gates=[5]), "a recovery targets no gate"),
    (lambda w: dict(start_step="1200"), "start_step must be a whole number"),
    (lambda w: dict(authorization=["words"]), "the recovery's authorization must be text"),
], ids=["stopped-step", "adoption-sha", "adoption-run", "code-hash", "code-block", "role", "repeats", "targets",
        "start-str", "authorization-text"])
def test_the_blocks_form(world, change, needle):
    w = world()
    _refused_written(w, needle, **change(w))


def test_the_noting_of_a_recovery_needs_src(world):
    w = world()
    _write(w)
    run = _folder(w, "run_a")
    for fn in (_note, _mark):
        with pytest.raises(TP.PreflightError, match="a recovery's attempt and launch are noted with src"):
            fn(w, run, src=False)


def test_set_recovery_needs_a_record_and_who_writes_it(world):
    w = world()
    with pytest.raises(TP.PreflightError, match="no record for configuration"):
        TP.set_recovery(OTHER_FP, _block(w), "tester", root=w["root"], registry=w["reg"], src=w["src"])
    with pytest.raises(TP.PreflightError, match="who writes it"):
        TP.set_recovery(w["fp"], _block(w), " ", root=w["root"], registry=w["reg"], src=w["src"])


def test_the_stop_criteria_are_checked_as_a_continuations(world):
    w = world()
    _refused_written(w, "no stop criterion has mode stop", stop_criteria=[dict(STOP_CRITERIA[0], mode="alert")])


def test_the_command_line_writes_a_recovery_and_marks_its_launch(world, monkeypatch, capsys):
    w = world()
    path = w["tmp"] / "recovery_block.json"
    path.write_text(json.dumps(_block(w)), encoding="utf-8-sig")
    set0 = TP.set_recovery
    monkeypatch.setattr(TP, "set_recovery", lambda f, b, by, *x, **k: set0(f, b, by, root=w["root"], **k))
    assert TP.main(["recovery", w["fp"], "--block-file", str(path), "--by", "tester"]) == 2
    assert "recovery needs --src" in capsys.readouterr().out
    assert TP.main(["recovery", w["fp"], "--block-file", str(path), "--src", str(w["src"]), "--registry",
                    str(w["reg"]), "--by", "tester"]) == 0
    assert "recovery written" in capsys.readouterr().out
    mark0 = TP.mark_launched
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: mark0(f, r, by, root=w["root"], **k))
    run = _folder(w, "run_cli")
    assert TP.main(["launched", w["fp"], "--run", str(run), "--recovery", "--adoption", "--by", "t"]) == 2
    assert "launched marks one block" in capsys.readouterr().out
    assert TP.main(["launched", w["fp"], "--run", str(run), "--recovery", "--registry", str(w["reg"]),
                    "--by", "tester"]) == 2
    assert "noted with src" in capsys.readouterr().out
    assert TP.main(["launched", w["fp"], "--run", str(run), "--recovery", "--src", str(w["src"]), "--registry",
                    str(w["reg"]), "--by", "tester"]) == 0
    assert "recovery marked launched" in capsys.readouterr().out
    assert _held(w)["block"]["run"] == str(run.resolve())
    assert _rows(w)["recover_stop_row"][TP.RECOVERY_ROW_FIELD]["launched_at"]


def test_the_command_lines_check_reads_another_checkout_registry_and_folder(world, monkeypatch, capsys):
    w = world()
    _write(w)
    seen = {}
    refusal0 = TP.launch_refusal
    monkeypatch.setattr(TP, "launch_refusal", lambda cmd, **k: seen.update(k) or refusal0(cmd, **k))
    path = w["tmp"] / "cmd_r.json"
    path.write_text(json.dumps(w["cmd_r"]), encoding="utf-8")
    assert TP.main(["check", "--cmd-file", str(path), "--src", str(w["src"]), "--registry", str(w["reg"]), "--root",
                    str(w["root"])]) == 0
    assert "ALLOWED: configuration %s may launch." % w["fp"] in capsys.readouterr().out
    assert seen == dict(src=w["src"], registry=w["reg"], root=w["root"])
    assert TP.main(["check", "--cmd-file", str(path), "--src", str(w["src"]), "--registry", str(w["reg"])]) == 1
    assert "has no preflight record" in capsys.readouterr().out          # production's folder by default: not this one
    assert TP.main(["fingerprint", "--cmd-file", str(path), "--src", str(w["src"])]) == 0
    assert capsys.readouterr().out.strip() == w["fp"]
