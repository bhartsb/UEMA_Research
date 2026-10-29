# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE LINKED CONTINUATION OF A STOPPED RECOVERY (training_preflight.py, the module text; proxy decision
proxy-m6-result-next-run-20261013 of Oct 13, 2026: "Prepare one bounded
continuation from the preserved complete 492800 state to the existing endpoint 494025 (1,225 model steps)", item 3 (the
one stop-policy change: finite proprioception M6 crossings an alert), items 4 and 5 (unchanged code and settings, the
old stop and INCOMPLETE kept, the remaining time from actual trainer lifetimes), and the coordinator's instruction: name
the used recovery entry, the stopped checkpoint, the decision, the code and settings, the endpoint and the remaining
wall budget; keep the earlier records and entries used and unchanged; refuse a wrong parent or checkpoint, changed flags
or code, a reused authorization, more model steps or a larger cap, folders inside the preservations and a second use).

The world of these tests is the recovery's stand-in world of test_preflight_recovery.py (its _world, _route, _block and
_write, unchanged), continued: the recovery written, noted and marked launched for its run folder, which then trained
from 1,200 to 1,600 and stopped (its args.json, stats, stage_done.json, end checkpoint and latent body folder); the
recovery's rows ended as the run owner ends them (training row failed, its runner's state STOPPED; completion row
failed); a preservation of the stopped recovery run's end (its MANIFEST naming the run's checkpoint at 1,600 as its
source); its stopped receipt, its INCOMPLETE outcome receipt and its run owner's log (the trainer's process created at
2,000.5 s, its end observed at 2,300.25 s); the link decision's message; the recovery's launch command (the link's
template). The adoption's and the recovery's stop criteria hold a second stop line judged from run-hour 0.25, so the
link's 400 model steps judge it only on the watcher's carried clock (the preserved runs' 2,240 training seconds). The
link: one bounded run from the preserved 1,600 to 2,000 within floor(2000 - 100.5 - 299.75) = 1,599 s. Nothing here
writes a production file, and no test reads the production registry (the fixture `no_production_registry`), Saved/ or
D:/UEMA_Research/external/D/UEMA_preserved. The test of the production pins reads only this module's constants; the test of the command line's
check without --root reads the preflight folder of the checkout that holds the tool and finds no record of the stand-in
configuration there.

What is held (each refusal checked when the link is written, at launch, and when an attempt or a launch is noted,
wherever that moment can see the condition):
  (1) the link admits exactly its launch; the record, the used adoption's and the used recovery's files and both earlier
      blocks' rows stay byte for byte as they were; the launch is recorded in the link's own row; used once;
  (2) WRONG PARENT: the used recovery's and adoption's files, their use, their runs, decisions, spans, caps, rows and
      receipts, the record, the lineage, another configuration;
  (3) WRONG CHECKPOINT: the original (not preserved) end, the recovery's or the adoption's start, the rollback, the
      model, the latent body state, the worlds companion, the MANIFEST's source, step and verdict, the preserved
      args.json, the stopped receipt, an outcome that is not INCOMPLETE;
  (4) CHANGED TRAINING CONFIGURATION: a flag, the checkout's code, a template flag, the stop criteria, the template,
      the code's commit and hash;
  (5) REUSED AUTHORIZATION: the recovery's and the adoption's decisions, words and times, another link's decision, the
      decision's message and the delegation, the decision's pinned time;
  (6) EXCESS STEPS OR TIME: --max-steps, the end, the bound, the wall cap and both trainers' receipts it is computed
      from;
  (7) THE GATES: the recovery's exceptions carried unchanged with their applicability; no PASS;
  (8) THE CARRIED CLOCK and THE STOP-POLICY EXCEPTION;
  (9) THE ROWS AND THE LAUNCH: the pinned training and completion rows, a noted attempt, the launch's folders (inside
      the preservations, inside an earlier run, an earlier run's folder name), the run flags, src for the noting; the
      command line.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import pytest

import test_preflight_recovery as R            # the recovery's stand-in world (its helpers, unchanged)
from mimo_bridge.tools import training_preflight as TP

LINK_STOP = 1600
T2_CREATED, T2_ENDED = 2000.5, 2300.25
WALL_L = int(math.floor(R.WALL_A - (R.T_ENDED - R.T_CREATED) - (T2_ENDED - T2_CREATED)))       # 1599
STATS_L = ((1300, 300.0), (1400, 600.0), (1500, 900.0), (1600, 1200.0))
CARRIED_S = 440.0 + 600.0 + 1200.0             # the preserved runs' newest training seconds: start, stop, stop2
DECISION_L = "proxy-link-test-20261013"
POLICY_WORDS = ("3. For this proposed continuation only, make FINITE proprioception M6 magnitude crossings alerts. (a "
                "stand-in for the decision's item 3)")
MESSAGE_L = "H84 PROXY DECISION: %s\n(a stand-in for the link decision's text)\n\n%s\n" % (DECISION_L, POLICY_WORDS)
AUTH_L = R.QUOTED + ("\nDelegation reference: the operator's delegation of Oct 13, 2026; proxy decision %s, by %s: "
                     "continue the stopped recovery's preserved end to the adoption's end." % (DECISION_L, R.AUTHOR))
AUTHORIZED_AT_L = "Oct 13, 2026 08:19:45 PM PDT"
WATCH = "docs/gain_of_function_v4/heads_all_slot_path_watch.py"
STOP_CRITERIA_L = [R.STOP_CRITERIA[0],
                   dict(name="vision_from_a_quarter_hour", path="causal_pe_per_modality.vision", op=">", line=0.04,
                        from_run_hour=0.25, mode="stop", note="S2")]
CK3_REL = "run/checkpoints/ckpt_step_1600/"
COMPANION3_REL = "episode_companion/ckpt_step_1600_89ab.pt.gz"
PRESERVED_L = {CK3_REL + "meta.json": json.dumps(dict(step=1600)).encode("utf-8"),
               CK3_REL + "model.safetensors": b"the stopped recovery run's weights at model step 1600",
               CK3_REL + "optimizer.pt": b"the stopped recovery run's optimizer state",
               "latent_body/latest.pt": b"the stopped recovery run's latent body state",
               COMPANION3_REL: b"the stopped recovery run's eight worlds"}
OTHER_FP = R.OTHER_FP
PRODUCTION_LINK = dict(
    fingerprint="eea1e8bc7f446fb6",
    record_sha256="e282bfc08fb4142b9b11780115898ebc85f526c466be40ad5b9da8c22e57d083",
    adoption_sha256="1d15cca55d5b85ea1f87079f753a4648636c3ba18d4d5600ddf06d99ff572280",
    original_wall_cap_s=20683,
    recovery_sha256="06d9fb15c5728dc28ed8abb847ba68b881030927e0372f8fe597a1be9cdfcd46",
    recovery_decision="proxy-adoption-recovery-489625-20261013",
    recovery_run="D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/run/B1R_skip13_489625",
    recovery_row="b1_recover_B1R_skip13_489625", recovery_start=489625, recovery_wall_cap_s=19931,
    checkpoint="D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/run/checkpoints/ckpt_step_492800",
    manifest="D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/MANIFEST.json",
    manifest_sha256="bde1568e7d9f14987dff906031b289a3021fe7a8f89859ab4cee908f7b6171b2",
    preserved_root="D:/UEMA_Research/external/D/UEMA_preserved", predecessor_run="D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/run",
    model_sha256="3e3b20270355c2d85db779be7ebb41168ff3bbede3582408f14ecd5e06f38b99",
    companion_sha256="4c91844c91971d06c42699ad252707787b176836728fcd9c2fabc935ba2317ec",
    latent_sha256="5b9bce03b670b93417c74ad0d692d18eafbcace5a359423f4cc8aa6e075769ef",
    stopped_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/receipts/recovery/arm_end.json",
    stopped_receipt_sha256="f55aec6ed3453eb3e78ec4dae53996b31d0f5050dbefd42c9b5e41a4e71d1727",
    outcome_receipt="D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/reading/outcome_receipt.json",
    outcome_receipt_sha256="0e2d7037cd63788b494573730a020fa7096a19b48e34d9221fb794f6dd215774",
    decision="proxy-m6-result-next-run-20261013",
    decision_message="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/mailbox/"
                     "proxy-m6-result-next-run-20261013/message.json",
    decision_text_sha256="49e9931da14da7b94ce78565573929453a620c8b7811dd8cc0b063787e488aa0",
    decision_author="the overseer",
    authorized_at="Oct 13, 2026 08:19:45 PM PDT",
    delegation_record="C:/Users/bhart/Documents/UEMA_devA_private/Saved/gof_coordination/"
                      "OPERATOR_PROXY_DELEGATION_2026-10-13.txt",
    delegation_words=R.PRODUCTION_ROUTE["delegation_words"],
    code_commit="641c931a67edf2988ec886ed15c93c376cf18f46",
    command_template="D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/command.json",
    command_template_sha256="c396abf2d54c8621c23a73ec7365dfa364e65d1acadfd5920a93a54da0e85b9c",
    run_row="b1_continue_B1K_skip13_492800", completion_row="b1_continue_outcome_494025",
    start=492800, end=494025, wall_cap_s=12338,
    trainers=(("D:/UEMA_Research/external/D/UEMA_scratch/b1_adopt_489350/receipts/owner.jsonl", 1791913178.843792, 1791913930.4329011),
              ("D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/receipts/owner.jsonl", 1791925811.7154384, 1791933404.5869603)),
    carried_runs=("D:/UEMA_Research/external/D/UEMA_preserved/body1_continuation_B1C_skip13_484675/run",
                  "D:/UEMA_Research/external/D/UEMA_preserved/body1_extension_B1E_skip13_485975/run",
                  "D:/UEMA_Research/external/D/UEMA_preserved/b1_adopt_B1A_skip13_489350/run",
                  "D:/UEMA_Research/external/D/UEMA_preserved/b1_recover_B1R_skip13_489625/run"),
    carried_s=19065.0122859478)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsha(p) -> str:
    return _sha(Path(p).read_bytes())


def _add_rows(reg: Path, **rows):
    data = json.loads(reg.read_text(encoding="utf-8"))
    data["rows"] += [dict(id=k, status=v) for k, v in rows.items()]
    reg.write_text(json.dumps(data), encoding="utf-8")


def _lworld(tmp, monkeypatch):
    """The recovery's stand-in world (test_preflight_recovery._world), its recovery written, launched and stopped at
    1,600, its end preserved, its receipts, its rows ended, and the link's inputs (the module text)."""
    w = R._world(tmp, monkeypatch, used=True)
    w["monkeypatch"] = monkeypatch
    monkeypatch.setattr(TP, "RECOVERY_ROUTES", (R._route(w),))
    R._write(w)
    r_run, r_lat, reg = w["r_run"], w["r_lat"], w["reg"]
    r_run.mkdir(parents=True)
    R._note(w, r_run)
    R._edit_rows(reg, "recover_stop_row", status="running")
    R._mark(w, r_run)
    # the recovery's trainer: trained 1,200 -> 1,600 and was stopped; its end checkpoint and latent body state
    args = dict(resume=str(w["ck2"]), max_steps=R.END, report_every=100)
    (r_run / "args.json").write_text(json.dumps(args), encoding="utf-8")
    R._stats(r_run, STATS_L)
    done = json.dumps(dict(ok=True, steps=LINK_STOP, finished_at=time.time() - 20))
    (r_run / "stage_done.json").write_text(done, encoding="utf-8")
    R._write_files(r_run, {k.replace("run/", "", 1): v for k, v in PRESERVED_L.items() if k.startswith(CK3_REL)})
    r_lat.mkdir()
    (r_lat / "latest.pt").write_bytes(PRESERVED_L["latent_body/latest.pt"])
    R._edit_rows(reg, "recover_stop_row", status="failed", last_run=dict(state="STOPPED"))
    R._edit_rows(reg, "recover_done_row", status="failed", last_run=dict(state="INVALID"))
    # the preservation of the stopped recovery run's end
    p3 = tmp / "preserved" / "stop2"
    R._write_files(p3, PRESERVED_L)
    run3, ck3 = p3 / "run", p3 / CK3_REL.rstrip("/")
    (run3 / "args.json").write_text(json.dumps(args), encoding="utf-8")
    (run3 / "stage_done.json").write_text(done, encoding="utf-8")
    R._stats(run3, STATS_L)
    companion = _sha(PRESERVED_L[COMPANION3_REL])
    ck_r = r_run / "checkpoints" / "ckpt_step_1600"
    m3_body = R._manifest(p3, PRESERVED_L, run_files=("args.json", "stage_done.json"), checkpoint=str(ck_r),
                          step=LINK_STOP, model_sha256=_sha(PRESERVED_L[CK3_REL + "model.safetensors"]),
                          companion_sha256=companion, companion_sha256_in_episode_state=companion, companion_equal=True)
    stopped2 = tmp / "arm_end_recovery.json"
    stopped2.write_text(json.dumps(dict(state="stopped", step=LINK_STOP, run=str(r_run), trainer_exit_code=0,
                                        preserved=dict(verdict="pass", step=LINK_STOP), preserved_checkpoint=str(ck3))),
                        encoding="utf-8")
    outcome2 = tmp / "outcome_recovery.json"
    outcome2.write_text(json.dumps(dict(outcome="[two linked runs] INCOMPLETE - the recovery run stopped at model step "
                                                "1600")), encoding="utf-8")
    owner2 = tmp / "owner_recovery.jsonl"
    owner2.write_text("".join(json.dumps(e) + "\n" for e in (
        dict(event="trainer started", arm="recovery", pid=2, created=T2_CREATED, at_s=T2_CREATED + 0.2),
        dict(event="trainer ended", arm="recovery", exit_code=0, at_s=T2_ENDED))), encoding="utf-8")
    _add_rows(reg, link_stop_row="written", link_done_row="queued")
    message_l = tmp / "message_l.json"
    message_l.write_text(json.dumps(dict(id=DECISION_L, text=MESSAGE_L, created_at="2026-10-14T03:19:45+00:00")),
                         encoding="utf-8")
    command_r = tmp / "recovery_command.json"          # the recovery's launch command: the link's template
    command_r.write_text(json.dumps(w["cmd_r"], indent=1), encoding="utf-8")
    l_run, l_lat = tmp / "new_run_l", tmp / "lat_out_l"
    cmd_l = TP.apply_sets(w["cmd_r"], [("--resume", str(ck3)), ("--out", str(l_run)),
                                       ("--checkpoint-storage", str(l_run / "checkpoints")),
                                       ("--latent-body-resume", str(p3 / "latent_body" / "latest.pt")),
                                       ("--latent-body-output", str(l_lat))])
    w.update(p3=p3, run3=run3, ck3=ck3, ck_r=ck_r, manifest3=p3 / "MANIFEST.json", manifest3_sha=_sha(m3_body),
             model3_sha=_sha(PRESERVED_L[CK3_REL + "model.safetensors"]), companion3_sha=companion,
             latent3=p3 / "latent_body" / "latest.pt", latent3_sha=_sha(PRESERVED_L["latent_body/latest.pt"]),
             stopped2=stopped2, outcome2=outcome2, owner2=owner2, message_l=message_l, command_r=command_r,
             l_run=l_run, l_lat=l_lat, cmd_l=cmd_l,
             recovery_sha=_fsha(TP.recovery_path(w["fp"], w["root"])),
             adoption_sha_now=_fsha(TP.adoption_path(w["fp"], w["root"])))
    return w


def _lroute(w, **changes):
    route = dict(fingerprint=w["fp"], record_sha256=w["record_sha"], adoption_sha256=w["adoption_sha_now"],
                 original_wall_cap_s=R.WALL_A, recovery_sha256=w["recovery_sha"], recovery_decision=R.DECISION_R,
                 recovery_run=str(w["r_run"]), recovery_row="recover_stop_row", recovery_start=R.STOP,
                 recovery_wall_cap_s=R.WALL_R, checkpoint=str(w["ck3"]), manifest=str(w["manifest3"]),
                 manifest_sha256=w["manifest3_sha"], preserved_root=str(w["tmp"] / "preserved"),
                 predecessor_run=str(w["run3"]), model_sha256=w["model3_sha"], companion_sha256=w["companion3_sha"],
                 latent_sha256=w["latent3_sha"], stopped_receipt=str(w["stopped2"]),
                 stopped_receipt_sha256=_fsha(w["stopped2"]), outcome_receipt=str(w["outcome2"]),
                 outcome_receipt_sha256=_fsha(w["outcome2"]), decision=DECISION_L, decision_message=str(w["message_l"]),
                 decision_text_sha256=_sha(MESSAGE_L.encode("utf-8")), decision_author=R.AUTHOR,
                 authorized_at=AUTHORIZED_AT_L, delegation_record=str(w["delegation"]), delegation_words=R.SENTENCES,
                 code_commit=R.TO_COMMIT, command_template=str(w["command_r"]),
                 command_template_sha256=_fsha(w["command_r"]), run_row="link_stop_row", completion_row="link_done_row",
                 start=LINK_STOP, end=R.END, wall_cap_s=WALL_L,
                 trainers=((str(w["owner_log"]), R.T_CREATED, R.T_ENDED), (str(w["owner2"]), T2_CREATED, T2_ENDED)),
                 carried_runs=(str(w["p1"] / "run"), str(w["run2"]), str(w["run3"])), carried_s=CARRIED_S,
                 stop_policy=dict(rule="M6", watch=WATCH, sense="proprioception", crossing="finite", mode="alert",
                                  decision_words=POLICY_WORDS))
    route.update(changes)
    return route


def _exceptions_l(w):
    carried = R._held(w)["block"]["exceptions"]
    return {n: dict({f: x[f] for f in TP.ADOPTION_EXCEPTION_FIELDS},
                    carried_from="recoveries/%s.json gate %s" % (w["fp"], n),
                    applicability="the stopped recovery run trained the same configuration from the recovery's start; "
                                  "gate %s was not measured again, so its failure applies to the preserved 1,600" % n)
            for n, x in carried.items()}


def _lblock(w, **more):
    b = dict(purpose="the stopped recovery's remainder, 1,600 to 2,000",
             predecessor=dict(recovery=str(TP.recovery_path(w["fp"], w["root"])), recovery_sha256=w["recovery_sha"],
                              recovery_decision=R.DECISION_R, recovery_run=str(w["r_run"]),
                              adoption=str(TP.adoption_path(w["fp"], w["root"])), adoption_sha256=w["adoption_sha_now"],
                              stopped_step=LINK_STOP, stopped_receipt=str(w["stopped2"]),
                              stopped_receipt_sha256=_fsha(w["stopped2"]), checkpoint=str(w["ck3"]),
                              manifest_path=str(w["manifest3"]), manifest_sha256=w["manifest3_sha"],
                              model_sha256=w["model3_sha"], companion_sha256=w["companion3_sha"], run=str(w["run3"]),
                              outcome_receipt=str(w["outcome2"]), outcome_receipt_sha256=_fsha(w["outcome2"])),
             code=dict(commit=R.TO_COMMIT[:9], code_hash=w["code_hash"]),
             decision=dict(id=DECISION_L, message_sha256=_sha(MESSAGE_L.encode("utf-8")), author=R.AUTHOR),
             authorization=AUTH_L, authorized_by=R.AUTHOR, authorized_at=AUTHORIZED_AT_L,
             start_checkpoint=str(w["ck3"]), rollback_checkpoint=str(w["ck3"]), start_step=LINK_STOP, end_step=R.END,
             max_model_steps=R.END - LINK_STOP, wall_cap_s=WALL_L,
             run_flags={"--latent-body-resume": str(w["latent3"]), "--latent-body-output": str(w["l_lat"])},
             stop_lines="S1 and S2, carried; M6 an alert for proprioception's finite crossings",
             stop_lines_row="link_stop_row", stop_criteria=STOP_CRITERIA_L, completion_jobs_row="link_done_row",
             reachability="from 1,600 the first full stats record comes at 1,700; on the carried clock both stop lines "
                          "are first judged up to 235 model steps in, before the end at 2,000",
             no_auto_continue="the owner starts the trainer with --max-steps 2000 and no supervisor",
             carried_clock=dict(runs=[str(w["p1"] / "run"), str(w["run2"]), str(w["run3"])], seconds=CARRIED_S,
                                basis="the preserved runs' newest training seconds, oldest first"),
             stop_policy_exception=dict(rule="M6", watch=WATCH, sense="proprioception", crossing="finite", mode="alert",
                                        decision_words=POLICY_WORDS,
                                        still_stops="non-finite outputs; every other sense's M6 crossing; every other rule",
                                        applicability="this link only"),
             exceptions=_exceptions_l(w))
    b.update(more)
    return b


def _lpred(w, **changes):
    return dict(_lblock(w)["predecessor"], **changes)


@pytest.fixture(autouse=True)
def no_production_registry(monkeypatch, tmp_path):
    """The tool's default registry (note_attempt and mark_launched read it when none is given) is a file that does not
    exist, so no test can write production's registry."""
    monkeypatch.setattr(TP, "REGISTRY", tmp_path / "no_registry_given.json")


@pytest.fixture
def world(tmp_path, monkeypatch):
    """world(name, **route_changes): the link's stand-in world in its own folder, LINK_ROUTES naming its link with
    `route_changes` applied (pytest's monkeypatch restores every route after the test)."""
    monkeypatch.setattr(R, "STOP_CRITERIA", STOP_CRITERIA_L)

    def _make(name="w", **changes):
        w = _lworld(tmp_path / name, monkeypatch)
        monkeypatch.setattr(TP, "LINK_ROUTES", (_lroute(w, **changes),))
        return w
    return _make


def _pin(w, **changes):
    w["monkeypatch"].setattr(TP, "LINK_ROUTES", (_lroute(w, **changes),))


def _write(w, **more):
    return TP.set_link(w["fp"], _lblock(w, **more), "tester", root=w["root"], registry=w["reg"], src=w["src"])


def _refusal(w, cmd=None):
    return TP.launch_refusal(cmd or w["cmd_l"], root=w["root"], src=w["src"], registry=w["reg"])


def _plan(w, cmd=None):
    return TP.launch_plan(cmd or w["cmd_l"], root=w["root"], src=w["src"])


def _folder(w, name):
    run = w["tmp"] / "experiments" / name
    run.mkdir(parents=True, exist_ok=True)
    return run


def _note(w, run, src=True):
    return TP.note_attempt(w["fp"], str(run), "launcher", root=w["root"], block=TP.LINK,
                           src=w["src"] if src else None, registry=w["reg"])


def _mark(w, run, src=True):
    return TP.mark_launched(w["fp"], str(run), "launcher", root=w["root"], block=TP.LINK,
                            src=w["src"] if src else None, registry=w["reg"])


def _held(w):
    return json.loads(TP.link_path(w["fp"], w["root"]).read_text(encoding="utf-8"))


def _edit_link(w, **changes):
    held = _held(w)
    held["block"].update(changes)
    TP._save(TP.link_path(w["fp"], w["root"]), held)


def _rows(w):
    return R._rows(w)


def _refused_written(w, needle, **more):
    """Writing the link (with `more` in its block) is refused with `needle`, and no link file is written."""
    with pytest.raises(TP.PreflightError) as exc:
        _write(w, **more)
    assert needle in str(exc.value), str(exc.value)
    assert not TP.link_path(w["fp"], w["root"]).exists()


def _refused_noted(w, needle):
    """Noting an attempt and marking a launch of the link are each refused with `needle`; nothing is recorded."""
    run = _folder(w, "run_noted")
    for fn in (_note, _mark):
        with pytest.raises(TP.PreflightError) as exc:
            fn(w, run)
        assert needle in str(exc.value), (fn.__name__, str(exc.value))
    assert not _held(w)["block"].get("attempts") and not _held(w)["block"].get("launched_at")
    assert not any(TP.LINK_ROW_FIELD in r for r in _rows(w).values())


def _refused_at_launch_and_noted(w, needle, cmd=None):
    why = _refusal(w, cmd)
    assert why and needle in why, why
    _refused_noted(w, needle)


def _refused_everywhere(world, needle, damage, name="x", **route_changes):
    """`damage` (applied to a fresh world before the link is written) refuses writing it; on a second world, applied
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
ATTEMPT = "the training row link_stop_row records this link's attempt"


def _kept(w):
    """The bytes of the record, the used adoption's and the used recovery's files, and both earlier blocks' rows."""
    return (TP.record_path(w["fp"], w["root"]).read_bytes(), TP.adoption_path(w["fp"], w["root"]).read_bytes(),
            TP.recovery_path(w["fp"], w["root"]).read_bytes(),
            {k: v for k, v in _rows(w).items() if k.startswith(("adopt_", "recover_"))})


# ------------------------------------------------------------------------------ (1) THE ADMITTED CASE
def test_the_link_admits_exactly_its_launch_and_keeps_every_earlier_record(world):
    w = world()
    before = _kept(w)
    assert _plan(w)["kind"] == "normal"                                 # no link yet
    b = _write(w)
    held = _held(w)
    assert held["fingerprint"] == w["fp"] and held["record_sha256"] == w["record_sha"]
    assert held["recovery_sha256"] == w["recovery_sha"] and held["adoption_sha256"] == w["adoption_sha_now"]
    assert held["route"]["checkpoint"] == str(w["ck3"]) and held["route"]["wall_cap_s"] == WALL_L == 1599
    assert b["role"] == "link" and b["written_at_s"] > 0
    assert _refusal(w) is None
    plan = _plan(w)
    assert (plan["kind"], plan["block"], plan["fingerprint"], plan["start_step"], plan["max_steps"],
            plan["wall_cap_s"]) == ("continuation", "link", w["fp"], LINK_STOP, R.END, WALL_L)
    assert plan["stop_criteria"] == STOP_CRITERIA_L
    run = _folder(w, "run_l")
    _note(w, run)
    R._edit_rows(w["reg"], "link_stop_row", status="running")           # the run owner's job_started
    _mark(w, run)
    rows = _rows(w)
    kept = rows["link_stop_row"][TP.LINK_ROW_FIELD]
    assert kept["fingerprint"] == w["fp"] and kept["decision"] == DECISION_L and kept["launched_at"]
    assert [a["run"] for a in kept["attempts"]] == [str(run.resolve())]
    assert TP.RECOVERY_ROW_FIELD not in rows["link_stop_row"] and TP.ADOPTION_ROW_FIELD not in rows["link_stop_row"]
    assert rows["link_done_row"] == dict(id="link_done_row", status="queued")              # only read
    # THE HISTORY: the record, the used adoption's and recovery's files and their rows byte for byte as they were
    assert _kept(w) == before
    rec = TP._load(TP.record_path(w["fp"], w["root"]))
    assert [g["status"] for g in rec["gates"]] == ["FAIL"] * 7 and [s["checkpoint"] for s in rec["states"]] == [
        str(w["ck1"])]
    rb = R._held(w)["block"]
    assert rb["launched_at"] and rb["run"] == str(w["r_run"].resolve())
    assert all({f: x[f] for f in TP.ADOPTION_EXCEPTION_FIELDS} == {f: rb["exceptions"][n][f]
                                                                    for f in TP.ADOPTION_EXCEPTION_FIELDS}
               for n, x in b["exceptions"].items())
    assert b["stop_policy_exception"]["sense"] == "proprioception" and b["stop_policy_exception"]["mode"] == "alert"


def test_the_link_is_used_once(world):
    w = world()
    _write(w)
    run = _folder(w, "run_x")
    _note(w, run)
    why = _refusal(w)
    assert why and ATTEMPT in why, why                                  # noted: no second launch, trained or not
    R._stats(run, ((1700, 108.0),))
    why = _refusal(w)
    assert why and LISTED in why and "and as its linked continuation (LINK_ROUTES)" in why, why
    assert _plan(w)["kind"] == "normal"
    with pytest.raises(TP.PreflightError, match="the link of configuration .* was used"):
        _write(w)
    _mark(w, run)
    with pytest.raises(TP.PreflightError, match="launched already"):
        _mark(w, run)
    with pytest.raises(TP.PreflightError, match="launched already"):
        _note(w, _folder(w, "run_y"))
    # neither earlier block can be written again: each stays used
    with pytest.raises(TP.PreflightError, match="the recovery of configuration .* was used"):
        R._write(w)
    with pytest.raises(TP.PreflightError, match="the adoption of configuration .* was used"):
        TP.set_adoption(w["fp"], R._adoption_block(w), "tester", root=w["root"], registry=w["reg"], src=w["src"])


def test_a_hand_edited_link_file_is_refused_by_its_training_row(world):
    w = world()
    _write(w)
    run = _folder(w, "run_first")
    _note(w, run)
    _mark(w, run)
    R._stats(run, ((1700, 108.0),))
    new_out = str(w["l_lat"]) + "_second"
    _edit_link(w, launched_at=None, launched_at_s=None, run=None, attempts=[],
               run_flags={"--latent-body-resume": str(w["latent3"]), "--latent-body-output": new_out})
    why = _refusal(w, TP.apply_sets(w["cmd_l"], [("--latent-body-output", new_out)]))
    assert why and "the training row link_stop_row records this link's launch marked" in why, why
    for fn in (_note, _mark):
        with pytest.raises(TP.PreflightError, match="records this link's launch marked"):
            fn(w, _folder(w, "run_second"))


def test_the_link_file_names_its_own_configuration(world):
    w = world()
    _write(w)
    held = _held(w)
    held["fingerprint"] = OTHER_FP
    TP._save(TP.link_path(w["fp"], w["root"]), held)
    _refused_at_launch_and_noted(w, "the link file names configuration %s, not %s" % (OTHER_FP, w["fp"]))


def test_a_link_file_written_against_another_record_is_refused(world):
    w = world()
    _write(w)
    held = _held(w)
    held["record_sha256"] = "0" * 64
    TP._save(TP.link_path(w["fp"], w["root"]), held)
    _refused_at_launch_and_noted(w, "the link file was written against the record sha256 %s; the link route pins %s" % (
        "0" * 64, w["record_sha"]))


def test_without_the_link_the_configuration_is_never_trained_from_the_stop(world):
    w = world()
    why = _refusal(w)
    assert why and LISTED in why and "and as its linked continuation (LINK_ROUTES), once, from %s to model step %d" % (
        w["ck3"], R.END) in why, why
    assert "and as its linked recovery (RECOVERY_ROUTES)" in why
    w["monkeypatch"].setattr(TP, "ADOPTION_ROUTES", ())
    w["monkeypatch"].setattr(TP, "RECOVERY_ROUTES", ())                  # a route listing the link alone
    why = _refusal(w)
    assert why and "is the linked continuation's (LINK_ROUTES): it launches only as its link" in why, why


def test_the_route_names_the_decisions_one_link():
    """The production pins: the values the decision names, compared with this module's own constants; the wall cap is
    the receipts' arithmetic (floor(20683 - (1791913930.4329011 - 1791913178.843792) - (1791933404.5869603 -
    1791925811.7154384)) = 12338); the recovery's cap is the first trainer's alone (19931); the carried clock ends with
    the stopped recovery run's preserved copy; no file is read."""
    assert len(TP.LINK_ROUTES) == 1
    route = dict(TP.LINK_ROUTES[0])
    policy = route.pop("stop_policy")
    assert route == PRODUCTION_LINK
    assert route["end"] - route["start"] == 1225
    lives = [ended - created for _log, created, ended in route["trainers"]]
    assert int(math.floor(route["original_wall_cap_s"] - sum(lives))) == route["wall_cap_s"] == 12338
    assert int(math.floor(route["original_wall_cap_s"] - lives[0])) == route["recovery_wall_cap_s"] == 19931
    rec = TP.RECOVERY_ROUTES[0]
    assert (route["fingerprint"], route["recovery_decision"], route["recovery_start"], route["end"],
            route["recovery_wall_cap_s"], route["recovery_row"], route["record_sha256"], route["adoption_sha256"],
            route["original_wall_cap_s"], route["code_commit"]) == (
        rec["fingerprint"], rec["decision"], rec["start"], rec["end"], rec["wall_cap_s"], rec["run_row"],
        rec["record_sha256"], rec["adoption_sha256"], rec["original_wall_cap_s"], rec["code_commit"])
    assert route["carried_runs"][-1] == route["predecessor_run"] and route["carried_runs"][-2] == rec["predecessor_run"]
    assert route["trainers"][0] == (rec["owner_log"], rec["trainer_created_s"], rec["trainer_exit_observed_s"])
    assert route["delegation_words"] == rec["delegation_words"] and route["decision"] != rec["decision"]
    assert (policy["rule"], policy["sense"], policy["crossing"], policy["mode"], policy["watch"]) == (
        "M6", "proprioception", "finite", "alert", "docs/gain_of_function_v4/heads_all_slot_path_watch.py")
    assert policy["decision_words"].startswith("3. For this proposed continuation only, make FINITE proprioception M6 "
                                               "magnitude crossings alerts.")
    assert "any other sense's simultaneous crossing must still stop" in policy["decision_words"]
    assert R.PRODUCTION_ROUTE == dict(TP.RECOVERY_ROUTES[0])            # the recovery's entry as reviewed


# ------------------------------------------------------------------------------ (2) WRONG PARENT
def test_a_used_recovery_file_whose_sha256_differs_is_refused(world):
    def damage(w):
        held = json.loads(TP.recovery_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
        held["block"]["note"] = "edited after its use"
        TP._save(TP.recovery_path(w["fp"], w["root"]), held)
    _refused_everywhere(world, "the used recovery's file", damage)


def test_a_used_adoption_file_whose_sha256_differs_is_refused(world):
    def damage(w):
        held = json.loads(TP.adoption_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
        held["block"]["note"] = "edited after its use"
        TP._save(TP.adoption_path(w["fp"], w["root"]), held)
    _refused_everywhere(world, "the used adoption's file", damage)


def test_a_recovery_that_was_never_used_is_refused(world):
    """The recovery file with its launch and attempts taken out, its new sha256 pinned: only the use is missing."""
    w = world()
    held = json.loads(TP.recovery_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
    held["block"].update(launched_at=None, launched_at_s=None, run=None, attempts=[])
    TP._save(TP.recovery_path(w["fp"], w["root"]), held)
    sha = _fsha(TP.recovery_path(w["fp"], w["root"]))
    _pin(w, recovery_sha256=sha)
    _refused_written(w, "was not used (no marked launch, no attempt that trained past its start): a link continues a "
                        "used recovery's stopped run", predecessor=_lpred(w, recovery_sha256=sha))


def test_an_adoption_launched_for_another_run_is_refused(world):
    """The adoption file's marked run changed, its new sha256 pinned: the adoption is not the one the recovery
    continued."""
    w = world()
    held = json.loads(TP.adoption_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
    held["block"]["run"] = str(w["tmp"] / "experiments" / "another_run")
    TP._save(TP.adoption_path(w["fp"], w["root"]), held)
    sha = _fsha(TP.adoption_path(w["fp"], w["root"]))
    _pin(w, adoption_sha256=sha)
    _refused_written(w, "the used adoption's marked launch names run %s; the used recovery continues the adoption run" % (
        w["tmp"] / "experiments" / "another_run"), predecessor=_lpred(w, adoption_sha256=sha))


def test_a_recovery_launched_for_another_run_is_refused(world):
    w = world()
    other = str(w["tmp"] / "experiments" / "another_run")
    _pin(w, recovery_run=other)
    _refused_written(w, "the used recovery's marked launch names run %s; the link route pins the run %s" % (
        str(w["r_run"].resolve()), other), predecessor=_lpred(w, recovery_run=other))


@pytest.mark.parametrize("key", ["recovery", "adoption"])
def test_a_block_naming_another_earlier_file_is_refused(world, key):
    w = world()
    other = str(w["tmp"] / "pre" / ("%ss" % key if key == "adoption" else "recoveries") / ("%s.json" % OTHER_FP))
    needle = "the link's predecessor names the used %s file %s" % (key, other)
    _refused_written(w, needle, predecessor=_lpred(w, **{key: other}))
    w2 = world("launch_" + key)
    _write(w2)
    _edit_link(w2, predecessor=_lpred(w2, **{key: other}))
    _refused_at_launch_and_noted(w2, needle)


def test_the_recoverys_decision_is_pinned(world):
    w = world(recovery_decision="another-recovery")
    _refused_written(w, "continues the recovery of decision another-recovery; the link names %s" % R.DECISION_R)
    _refused_written(w, "the used recovery names decision %s; the link route continues the recovery of decision "
                        "another-recovery" % R.DECISION_R, predecessor=_lpred(w, recovery_decision="another-recovery"))


def test_a_record_judged_again_since_the_recovery_is_refused(world):
    def damage(w):
        time.sleep(0.01)
        TP.set_gate(w["fp"], 4, "FAIL", "judged again", ["row_4b"], "tester", root=w["root"])
    _refused_everywhere(world, "(the link route's pin): a gate judged again or a state added since then is not carried "
                               "by the link", damage)


def test_the_recovery_was_written_against_the_pinned_record(world):
    w = world(record_sha256="7" * 64)
    _refused_written(w, "the used recovery was written against the record sha256 %s; the link route pins %s" % (
        w["record_sha"], "7" * 64))


def test_a_configuration_the_route_does_not_name_is_refused(world):
    w = world(fingerprint=OTHER_FP)
    _refused_written(w, "configuration %s is not one" % w["fp"])
    _pin(w)
    _write(w)
    _pin(w, fingerprint=OTHER_FP)
    _refused_noted(w, "configuration %s is not one" % w["fp"])


def test_a_start_whose_lineage_does_not_reach_the_recoverys_start_is_refused(world):
    """The preserved run's args.json resumes another checkpoint (its MANIFEST recording that file)."""
    w = world()
    (w["run3"] / "args.json").write_text(json.dumps(dict(resume=str(w["tmp"] / "else" / "run" / "checkpoints" /
                                                                   "ckpt_step_1200"), max_steps=R.END)), encoding="utf-8")
    body = R._manifest(w["p3"], PRESERVED_L, run_files=("args.json", "stage_done.json"), checkpoint=str(w["ck_r"]),
                       step=LINK_STOP, model_sha256=w["model3_sha"], companion_sha256=w["companion3_sha"],
                       companion_sha256_in_episode_state=w["companion3_sha"], companion_equal=True)
    _pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "does not continue the model state %s the used recovery started from" % w["ck2"],
                     predecessor=_lpred(w, manifest_sha256=_sha(body)))


@pytest.mark.parametrize("row, changes, needle", [
    ("recover_stop_row", dict(status="queued"), "the used recovery's training row recover_stop_row reads status queued"),
    ("recover_stop_row", dict(last_run=dict(state="PASS")), "and its runner's state PASS"),
    ("recover_stop_row", {TP.RECOVERY_ROW_FIELD: None}, "the used recovery's training row recover_stop_row does not "
                                                        "record its launch of run"),
    ("recover_done_row", dict(status="done"), "the used recovery's completion row recover_done_row reads status done"),
    ("adopt_stop_row", dict(status="queued"), "the used adoption's training row adopt_stop_row reads status queued"),
    ("adopt_stop_row", {TP.ADOPTION_ROW_FIELD: None}, "the used adoption's training row adopt_stop_row does not record "
                                                      "its launch of run"),
    ("adopt_done_row", dict(status="done"), "the used adoption's completion row adopt_done_row reads status done"),
], ids=["recovery-requeued", "recovery-pass", "recovery-record-removed", "recovery-completion-done",
        "adoption-requeued", "adoption-record-removed", "adoption-completion-done"])
def test_the_earlier_blocks_rows_keep_their_ends(world, row, changes, needle):
    _refused_everywhere(world, needle, lambda w: R._edit_rows(w["reg"], row, **changes))


@pytest.mark.parametrize("row, needle", [
    ("recover_stop_row", "the used recovery's training row recover_stop_row is not in the test registry"),
    ("adopt_stop_row", "the used adoption's training row adopt_stop_row is not in the test registry"),
    ("recover_done_row", "the used recovery's completion row recover_done_row is not in the test registry"),
    ("adopt_done_row", "the used adoption's completion row adopt_done_row is not in the test registry"),
], ids=["recovery-training", "adoption-training", "recovery-completion", "adoption-completion"])
def test_the_earlier_blocks_rows_must_exist(world, row, needle):
    w = world()
    data = json.loads(w["reg"].read_text(encoding="utf-8"))
    data["rows"] = [r for r in data["rows"] if r["id"] != row]
    w["reg"].write_text(json.dumps(data), encoding="utf-8")
    _refused_written(w, needle)


@pytest.mark.parametrize("which, needle", [
    ("stopped", "the used adoption's stopped receipt"),
    ("outcome_r", "the used adoption's outcome receipt"),
], ids=["adoption-stopped-receipt", "adoption-outcome-receipt"])
def test_the_adoptions_receipts_read_as_the_recovery_names_them(world, which, needle):
    """A byte added to the adoption run's stopped or outcome receipt: it no longer reads the sha256 the used recovery
    names for it."""
    def damage(w):
        w[which].write_bytes(w[which].read_bytes() + b"\n")
    _refused_everywhere(world, needle, damage)
    w = world("named")
    w[which].write_bytes(w[which].read_bytes() + b"\n")
    with pytest.raises(TP.PreflightError) as exc:
        _write(w)
    assert "%s %s reads sha256" % (needle, w[which]) in str(exc.value) and "; the used recovery names sha256" in str(
        exc.value), str(exc.value)


def test_the_adoptions_outcome_stays_INCOMPLETE(world):
    """The adoption's outcome receipt rewritten as a success and the recovery file pinned to name it: a successful
    predecessor is never made."""
    w = world()
    w["outcome_r"].write_text(json.dumps(dict(outcome="A. CANDIDATE FOR ADOPTION")), encoding="utf-8")
    held = json.loads(TP.recovery_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
    held["block"]["predecessor"]["outcome_receipt_sha256"] = _fsha(w["outcome_r"])
    TP._save(TP.recovery_path(w["fp"], w["root"]), held)
    sha = _fsha(TP.recovery_path(w["fp"], w["root"]))
    _pin(w, recovery_sha256=sha)
    _refused_written(w, "the used adoption's outcome receipt %s reads 'A. CANDIDATE FOR ADOPTION'; that run's result "
                        "is INCOMPLETE" % w["outcome_r"], predecessor=_lpred(w, recovery_sha256=sha))


@pytest.mark.parametrize("changes, needle", [
    (dict(recovery_start=R.STOP + 1), "the used recovery ran from model step %d to %d; the link route continues the "
                                      "recovery from %d" % (R.STOP, R.END, R.STOP + 1)),
    (dict(recovery_row="other_row"), "the used recovery's training row reads 'recover_stop_row'; the link route pins "
                                     "other_row"),
    (dict(code_commit="9" * 40), "the used recovery ran the code of commit %r; the link runs the code of commit %s" % (
        R.TO_COMMIT[:9], "9" * 40)),
], ids=["recovery-start", "recovery-row", "code-commit"])
def test_the_recoverys_span_row_and_code_are_pinned(world, changes, needle):
    w = world(**changes)
    more = dict(code=dict(commit="9" * 40, code_hash=w["code_hash"])) if "code_commit" in changes else {}
    _refused_written(w, needle, **more)


def test_the_recoverys_cap_is_pinned(world):
    """A recovery cap the route pins against the first trainer's arithmetic and the recovery file's cap: both refuse."""
    w = world(recovery_wall_cap_s=R.WALL_R + 1)
    _refused_written(w, "the used recovery's wall cap reads %d; the link route pins %d s" % (R.WALL_R, R.WALL_R + 1))
    _refused_written(w, "the link route's recovery cap %d s is not the adoption's %d s less the first trainer's 100.500 "
                        "s" % (R.WALL_R + 1, R.WALL_A))


def test_the_adoptions_original_cap_is_pinned(world):
    w = world(original_wall_cap_s=R.WALL_A + 1)
    _refused_written(w, "the used adoption's wall cap reads %d; the link's remaining cap is computed from the original "
                        "%d s" % (R.WALL_A, R.WALL_A + 1))


# ------------------------------------------------------------------------------ (3) WRONG CHECKPOINT
def test_the_original_end_checkpoint_instead_of_its_preserved_copy_is_refused(world):
    w = world()
    _refused_written(w, "admits only the stopped recovery run's preserved end checkpoint",
                     start_checkpoint=str(w["ck_r"]), rollback_checkpoint=str(w["ck_r"]),
                     predecessor=_lpred(w, checkpoint=str(w["ck_r"])))


@pytest.mark.parametrize("start", ["ck2", "ck1"], ids=["the-recoverys-start", "the-adoptions-start"])
def test_an_earlier_start_is_refused(world, start):
    w = world()
    step = R.STOP if start == "ck2" else R.START_A
    _refused_written(w, "admits only the stopped recovery run's preserved end checkpoint", start_checkpoint=str(w[start]),
                     rollback_checkpoint=str(w[start]), predecessor=_lpred(w, checkpoint=str(w[start])),
                     start_step=step, max_model_steps=R.END - step)


def test_the_rollback_checkpoint_is_the_start(world):
    w = world()
    _refused_written(w, "the link's rollback checkpoint reads %s" % w["ck2"], rollback_checkpoint=str(w["ck2"]))


def test_a_launch_from_another_checkpoint_is_not_the_link(world):
    w = world()
    _write(w)
    for ck in (w["ck_r"], w["ck2"]):
        cmd = TP.apply_sets(w["cmd_l"], [("--resume", str(ck))])
        why = _refusal(w, cmd)
        assert why and LISTED in why, why
        assert _plan(w, cmd)["kind"] == "normal"
    cmd = TP.apply_sets(w["cmd_l"], [("--resume", str(w["ck_r"]))])
    why = TP._link_refusal(cmd, TP._load(TP.record_path(w["fp"], w["root"])), _held(w),
                           TP.configuration(cmd, w["src"]), w["root"], w["reg"])
    assert why and "not the link's start" in why, why


L_PIN_CHANGES = [
    ("checkpoint", lambda w: str(w["ck_r"]), "admits only the stopped recovery run's preserved end checkpoint"),
    ("manifest", lambda w: str(w["tmp"] / "elsewhere" / "MANIFEST.json"), "pins the preservation MANIFEST"),
    ("predecessor_run", lambda w: str(w["tmp"] / "elsewhere" / "run"), "pins the preserved run"),
    ("recovery_run", lambda w: str(w["tmp"] / "elsewhere" / "run"), "pins the used recovery's run"),
    ("stopped_receipt", lambda w: str(w["tmp"] / "elsewhere" / "arm_end.json"), "pins the stopped receipt"),
    ("outcome_receipt", lambda w: str(w["tmp"] / "elsewhere" / "outcome.json"), "pins the outcome receipt"),
    ("manifest_sha256", lambda w: "0" * 64, "pins the preservation MANIFEST's sha256 " + "0" * 64),
    ("model_sha256", lambda w: "1" * 64, "pins the model's sha256 " + "1" * 64),
    ("companion_sha256", lambda w: "2" * 64, "pins the worlds companion's sha256 " + "2" * 64),
    ("outcome_receipt_sha256", lambda w: "3" * 64, "pins the outcome receipt's sha256 " + "3" * 64),
    ("stopped_receipt_sha256", lambda w: "4" * 64, "pins the stopped receipt's sha256 " + "4" * 64),
    ("recovery_sha256", lambda w: "5" * 64, "pins the used recovery file's sha256 " + "5" * 64),
    ("adoption_sha256", lambda w: "a" * 64, "pins the used adoption file's sha256 " + "a" * 64),
    ("latent_sha256", lambda w: "6" * 64, "pins the latent body state's sha256 " + "6" * 64),
    ("decision", lambda w: "another-decision", "pins the decision another-decision; the link names " + DECISION_L),
    ("decision_text_sha256", lambda w: "8" * 64, "pins the decision message's text sha256 " + "8" * 64),
    ("decision_author", lambda w: "someone else", "pins the decision's author 'someone else'"),
    ("authorized_at", lambda w: "Oct 13, 2026 08:20:00 PM PDT", "pins the decision's time 'Oct 13, 2026 08:20:00 PM PDT'"),
    ("start", lambda w: 1599, "continues the recovery run stopped at model step 1599"),
    ("end", lambda w: 1900, "ends at model step 1900; the link's end_step reads 2000"),
    ("wall_cap_s", lambda w: 1598, "pins the remaining wall cap 1598 s; the link's wall_cap_s reads 1599"),
    ("run_row", lambda w: "other_row", "pins the training row other_row; the link's stop_lines_row reads "
                                       "'link_stop_row'"),
    ("completion_row", lambda w: "other_row", "pins the completion row other_row; the link's completion_jobs_row reads "
                                              "'link_done_row'"),
    ("code_commit", lambda w: "9" * 40, "runs the trainer code of commit " + "9" * 40),
]


@pytest.mark.parametrize("key, value, needle", L_PIN_CHANGES, ids=[c[0] for c in L_PIN_CHANGES])
def test_each_pinned_value_is_compared_exactly(world, key, value, needle):
    w = world("written")
    _pin(w, **{key: value(w)})
    _refused_written(w, needle)
    w2 = world("launch")
    _write(w2)
    _pin(w2, **{key: value(w2)})
    _refused_at_launch_and_noted(w2, needle)


def test_a_predecessor_model_whose_sha256_differs_is_refused(world):
    _refused_everywhere(world, "the predecessor's model", lambda w: (w["ck3"] / "model.safetensors").write_bytes(b"x"))


def test_a_latent_body_state_other_than_the_stopped_runs_is_refused(world):
    """The recovery's start's latent body state (the preserved 1,200), not the stopped recovery run's."""
    w = world()
    other = w["latent2"]
    _refused_written(w, "pins the latent body state's sha256 %s; the state the link resumes, %s, reads sha256 %s" % (
        w["latent3_sha"], other, _fsha(other)),
        run_flags={"--latent-body-resume": str(other), "--latent-body-output": str(w["l_lat"])})


@pytest.mark.parametrize("changes, needle", [
    (dict(companion_sha256="e" * 64), "records the worlds companion %s" % ("e" * 64)),
    (dict(companion_equal=False), "equal: False"),
    (dict(companion_sha256_in_episode_state="e" * 64), "names %s, equal" % ("e" * 64)),
    (dict(checkpoint="E:/another/run/checkpoints/ckpt_step_1600"), "records the source checkpoint "
                                                                   "E:/another/run/checkpoints/ckpt_step_1600"),
    (dict(step=1500), "at model step 1500"),
    (dict(verdict="fail"), "reads verdict fail; a link starts only from a preservation that passed"),
], ids=["companion", "companion-equal", "companion-in-episode-state", "source", "step", "verdict"])
def test_the_preservation_names_the_stopped_runs_end_and_its_worlds(world, changes, needle):
    w = world()
    fields = dict(checkpoint=str(w["ck_r"]), step=LINK_STOP, model_sha256=w["model3_sha"],
                  companion_sha256=w["companion3_sha"], companion_sha256_in_episode_state=w["companion3_sha"],
                  companion_equal=True)
    fields.update(changes)
    body = R._manifest(w["p3"], PRESERVED_L, run_files=("args.json", "stage_done.json"), **fields)
    _pin(w, manifest_sha256=_sha(body))
    _refused_written(w, needle, predecessor=_lpred(w, manifest_sha256=_sha(body)))


def test_a_companion_file_other_than_the_pinned_is_refused(world):
    w = world()
    other = {k: (b"other worlds" if k == COMPANION3_REL else v) for k, v in PRESERVED_L.items()}
    R._write_files(w["p3"], other)
    body = R._manifest(w["p3"], other, run_files=("args.json", "stage_done.json"), checkpoint=str(w["ck_r"]),
                       step=LINK_STOP, model_sha256=w["model3_sha"], companion_sha256=w["companion3_sha"],
                       companion_sha256_in_episode_state=w["companion3_sha"], companion_equal=True)
    _pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "companion files ['%s']" % _sha(b"other worlds"),
                     predecessor=_lpred(w, manifest_sha256=_sha(body)))


def test_the_preserved_args_json_is_the_one_the_manifest_records(world):
    def edited(w):
        args = json.loads((w["run3"] / "args.json").read_text(encoding="utf-8"))
        (w["run3"] / "args.json").write_text(json.dumps(dict(args, note="edited")), encoding="utf-8")
    _refused_everywhere(world, "the preserved run's args.json", edited)


def test_a_preserved_run_launched_for_another_end_is_refused(world):
    w = world()
    (w["run3"] / "args.json").write_text(json.dumps(dict(resume=str(w["ck2"]), max_steps=1800)), encoding="utf-8")
    body = R._manifest(w["p3"], PRESERVED_L, run_files=("args.json", "stage_done.json"), checkpoint=str(w["ck_r"]),
                       step=LINK_STOP, model_sha256=w["model3_sha"], companion_sha256=w["companion3_sha"],
                       companion_sha256_in_episode_state=w["companion3_sha"], companion_equal=True)
    _pin(w, manifest_sha256=_sha(body))
    _refused_written(w, "and ends at 1800; the used recovery started from", predecessor=_lpred(
        w, manifest_sha256=_sha(body)))


@pytest.mark.parametrize("changes, needle", [
    (dict(state="completed", step=R.END), "reads state completed at model step 2000"),
    (dict(trainer_exit_code=1), "trainer exit code 1"),
    (dict(preserved=dict(verdict="fail")), "preservation fail"),
    (dict(run="E:/another/run"), "(run E:/another/run"),
    (dict(preserved_checkpoint="E:/another/ckpt_step_1600"), "at E:/another/ckpt_step_1600"),
], ids=["completed", "exit-code", "preservation", "run", "preserved-checkpoint"])
def test_the_stopped_receipt_reads_a_stop_at_the_start_with_its_end_preserved(world, changes, needle):
    """A run that reached its end, or whose stop was not cleanly saved and preserved, leaves nothing for a link: it is
    never presented as a stopped predecessor (no successful predecessor is made)."""
    w = world()
    data = json.loads(w["stopped2"].read_text(encoding="utf-8"))
    data.update(changes)
    w["stopped2"].write_text(json.dumps(data), encoding="utf-8")
    sha = _fsha(w["stopped2"])
    _pin(w, stopped_receipt_sha256=sha)
    _refused_written(w, needle, predecessor=_lpred(w, stopped_receipt_sha256=sha))


def test_a_stopped_receipt_whose_sha256_differs_is_refused(world):
    w = world()
    named = _fsha(w["stopped2"])
    w["stopped2"].write_bytes(w["stopped2"].read_bytes() + b"\n")
    _refused_written(w, "the stopped receipt %s reads sha256" % w["stopped2"],
                     predecessor=_lpred(w, stopped_receipt_sha256=named))
    w2 = world("launch")
    _write(w2)
    w2["stopped2"].write_bytes(w2["stopped2"].read_bytes() + b"\n")
    _refused_at_launch_and_noted(w2, "the stopped receipt %s reads sha256" % w2["stopped2"])


def test_a_manifest_that_changed_after_the_link_is_refused(world):
    w = world()
    _write(w)
    w["manifest3"].write_bytes(w["manifest3"].read_bytes() + b"\n")
    _refused_at_launch_and_noted(w, "(the preservation record changed, or it is another)")


def test_an_outcome_that_is_not_INCOMPLETE_is_refused(world):
    w = world()
    w["outcome2"].write_text(json.dumps(dict(outcome="A. CANDIDATE FOR ADOPTION")), encoding="utf-8")
    sha = _fsha(w["outcome2"])
    _pin(w, outcome_receipt_sha256=sha)
    _refused_written(w, "that run's result is INCOMPLETE and the link keeps it",
                     predecessor=_lpred(w, outcome_receipt_sha256=sha))


def test_an_outcome_receipt_whose_sha256_differs_is_refused(world):
    w = world()
    named = _fsha(w["outcome2"])
    w["outcome2"].write_bytes(w["outcome2"].read_bytes() + b"\n")
    _refused_written(w, "the outcome receipt %s reads sha256" % w["outcome2"],
                     predecessor=_lpred(w, outcome_receipt_sha256=named))
    w2 = world("launch")
    _write(w2)
    w2["outcome2"].write_bytes(w2["outcome2"].read_bytes() + b"\n")
    _refused_at_launch_and_noted(w2, "the outcome receipt %s reads sha256" % w2["outcome2"])


def test_a_preserved_run_that_did_not_end_at_the_stop_is_refused(world):
    def damage(w):
        (w["run3"] / "stage_done.json").write_text(json.dumps(dict(ok=True, steps=R.END, finished_at=1.0)),
                                                   encoding="utf-8")
    _refused_everywhere(world, "did not end with its own save at model step 1600", damage)


def test_every_preserved_file_is_hashed_when_written_and_at_launch(world):
    w = world()
    (w["ck3"] / "optimizer.pt").write_bytes(b"other bytes")
    _refused_written(w, "the preserved file %s" % (w["p3"] / CK3_REL / "optimizer.pt"))
    w2 = world("launch")
    _write(w2)
    (w2["p3"] / COMPANION3_REL).write_bytes(b"other worlds")
    why = _refusal(w2)
    assert why and "the preserved file %s" % (w2["p3"] / COMPANION3_REL) in why, why


# ------------------------------------------------------------------------------ (4) CHANGED TRAINING CONFIGURATION
def test_a_command_whose_configuration_differs_is_refused_naming_each_difference(world):
    w = world()
    _write(w)
    flag = TP.apply_sets(w["cmd_l"], [("--proprio-token-skip", "3")])
    why = _refusal(w, flag)
    assert why and "is not the link's; a link runs exactly its record's code and settings" in why, why
    assert "flag --proprio-token-skip: 1,3 -> 3" in why
    (w["src"] / "lem" / "model" / "core.py").write_text("x = 2\n", encoding="utf-8")
    why = _refusal(w)
    assert why and "is not the link's" in why and "code: 1 file(s) differ: lem/model/core.py" in why, why
    # once the link is used, the same commands are judged exactly as before (no record of their configuration)
    (w["src"] / "lem" / "model" / "core.py").write_text("x = 1\n", encoding="utf-8")
    _mark(w, _folder(w, "run_used"))
    why = _refusal(w, flag)
    assert why and "has no preflight record" in why and "is not the link's" not in why, why


def test_another_launch_from_another_checkpoint_is_judged_as_before(world):
    w = world()
    _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd_l"], [("--proprio-token-skip", "3"), ("--resume", str(w["ck_r"]))]))
    assert why and "has no preflight record" in why and "is not the link's" not in why, why


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
def test_every_flag_but_the_replaced_ones_reads_as_the_recoverys_launch(world, sets):
    w = world()
    _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd_l"], sets))
    assert why and "the launch's %s reads %s; the command template reads" % sets[0] in why, why


def test_the_stop_criteria_are_the_recoverys(world):
    w = world()
    changed = [STOP_CRITERIA_L[0], dict(STOP_CRITERIA_L[1], mode="alert")]
    _refused_written(w, "the link's stop criteria are not the used recovery's", stop_criteria=changed)
    w2 = world("launch")
    _write(w2)
    _edit_link(w2, stop_criteria=changed)
    _refused_at_launch_and_noted(w2, "the link's stop criteria are not the used recovery's")


def test_the_command_template_reads_its_pinned_sha256(world):
    w = world()
    w["command_r"].write_text(w["command_r"].read_text(encoding="utf-8") + " ", encoding="utf-8")
    _refused_written(w, "the command template %s reads sha256" % w["command_r"])


def test_the_blocks_code_hash_is_the_records(world):
    w = world()
    _refused_written(w, "the link's code names code hash %s" % ("c" * 64), code=dict(commit=R.TO_COMMIT,
                                                                                    code_hash="c" * 64))


def test_a_record_whose_settings_were_edited_is_refused(world):
    w = world()
    rec = TP._load(TP.record_path(w["fp"], w["root"]))
    rec["code_hash"] = "0" * 64
    TP._save(TP.record_path(w["fp"], w["root"]), rec)
    _refused_written(w, "its settings or code were changed after it was written")


# ------------------------------------------------------------------------------ (5) REUSED AUTHORIZATION
@pytest.mark.parametrize("decision, words, whose", [
    (R.DECISION_R, R.AUTH_R, "recovery"), (R.DECISION_A, R.AUTH_A, "adoption")], ids=["recovery", "adoption"])
def test_an_earlier_blocks_decision_is_not_the_links(world, decision, words, whose):
    w = world(decision=decision)
    with pytest.raises(TP.PreflightError) as exc:
        _write(w, decision=dict(id=decision, message_sha256=_sha(MESSAGE_L.encode("utf-8")), author=R.AUTHOR),
               authorization=AUTH_L.replace(DECISION_L, decision))
    assert "the decision %s already authorized the %s" % (decision, whose) in str(exc.value), str(exc.value)


@pytest.mark.parametrize("words", [R.AUTH_R, R.AUTH_A], ids=["recovery", "adoption"])
def test_an_earlier_blocks_words_are_refused(world, words):
    _refused_written(world(), "the link's authorization and the authorization words of the",
                     authorization=words + "\n" + DECISION_L)


@pytest.mark.parametrize("whose", ["recovery", "adoption"])
def test_an_earlier_blocks_authorized_time_is_refused(world, whose):
    w = world()
    theirs = (R._held(w) if whose == "recovery" else {"block": R._adoption_block(w)})["block"]["authorized_at"]
    _pin(w, authorized_at=theirs)
    _refused_written(w, "the link's authorized_at repeats the used %s's" % whose, authorized_at=theirs)


def test_a_decision_already_used_by_another_link_is_refused(world):
    def damage(w):
        TP._save(TP.link_path(OTHER_FP, w["root"]), dict(fingerprint=OTHER_FP, block=dict(
            authorization="other words", decision=dict(id=DECISION_L, message_sha256="f" * 64))))
    _refused_everywhere(world, "the decision %s already authorized the link" % DECISION_L, damage)


def test_a_decision_message_already_used_is_refused(world):
    w = world()
    TP._save(TP.link_path(OTHER_FP, w["root"]), dict(fingerprint=OTHER_FP, block=dict(
        authorization="other words", decision=dict(id="another", message_sha256=_sha(MESSAGE_L.encode("utf-8"))))))
    _refused_written(w, "the decision message whose sha256 reads %s already authorized" % _sha(MESSAGE_L.encode("utf-8")))


def test_a_link_file_is_among_the_blocks_a_recovery_is_compared_with(world):
    """The authorization scan reads the links folder: a recovery written with a link's decision is refused."""
    w = world()
    _write(w)
    found = [where for where, _b in TP._authorized_blocks(w["root"])]
    assert any("the link" in where for where in found), found


def test_authorization_words_that_do_not_name_the_decision_are_refused(world):
    _refused_written(world(), "the link's authorization does not name its decision %s" % DECISION_L,
                     authorization=R.QUOTED)


@pytest.mark.parametrize("damage, needle", [
    (lambda w: w["message_l"].write_text(json.dumps(dict(id=DECISION_L, text=MESSAGE_L + " edited")),
                                         encoding="utf-8"), "holds a text whose sha256 reads"),
    (lambda w: w["delegation"].write_text("nothing", encoding="utf-8"), "does not hold the operator's sentence"),
], ids=["message-text", "delegation"])
def test_the_decision_message_and_the_delegation_record_are_read(world, damage, needle):
    _refused_everywhere(world, needle, damage)


def test_the_authorization_holds_each_delegation_sentence_verbatim(world):
    _refused_written(world(), "the link's authorization does not hold the operator's delegation sentence",
                     authorization=AUTH_L.replace(R.SENTENCES[1], R.SENTENCES[1].lower()))


# ------------------------------------------------------------------------------ (6) EXCESS STEPS OR TIME
def test_the_launch_ends_exactly_at_the_adoptions_end(world):
    w = world()
    _write(w)
    for steps in (str(R.END + 1), str(R.END - 1), str(LINK_STOP)):
        why = _refusal(w, TP.apply_sets(w["cmd_l"], [("--max-steps", steps)]))
        assert why and "the launch is not bounded as declared: --max-steps reads %s" % steps in why, why
    bare = list(w["cmd_l"])
    del bare[bare.index("--max-steps"):bare.index("--max-steps") + 2]
    why = _refusal(w, bare)
    assert why and "--max-steps reads absent" in why, why


def test_an_end_beyond_the_adoptions_is_refused(world):
    _refused_written(world(), "ends at model step 2000; the link's end_step reads 2001", end_step=R.END + 1,
                     max_model_steps=R.END + 1 - LINK_STOP)


def test_max_model_steps_other_than_end_minus_start_is_refused(world):
    needle = "max_model_steps 401 is not its end_step minus its start_step (2000 - 1600 = 400)"
    w = world()
    _refused_written(w, needle, max_model_steps=401)
    _write(w)
    _edit_link(w, max_model_steps=401)
    _refused_at_launch_and_noted(w, needle)


@pytest.mark.parametrize("cap", [WALL_L + 1, R.WALL_R, R.WALL_A, 1.0e9],
                         ids=["one-second-more", "the-recoverys-cap", "the-adoptions-cap", "huge"])
def test_a_wall_cap_above_the_remaining_time_is_refused(world, cap):
    _refused_written(world(), "pins the remaining wall cap %d s; the link's wall_cap_s reads" % WALL_L, wall_cap_s=cap)


@pytest.mark.parametrize("cap", [0, -1.0, "1599", float("nan"), True], ids=["zero", "negative", "text", "nan", "bool"])
def test_the_wall_cap_is_a_number_of_seconds_above_zero(world, cap):
    _refused_written(world(), "wall_cap_s must be a number of seconds above 0", wall_cap_s=cap)


@pytest.mark.parametrize("log, events, needle", [
    ("owner2", [dict(event="trainer started", created=T2_CREATED)], "records none for its 'trainer ended' event(s)"),
    ("owner2", [dict(event="trainer started", created=T2_CREATED), dict(event="trainer ended", at_s=T2_ENDED),
                dict(event="trainer ended", at_s=T2_ENDED + 60)],
     "records [%r, %r] for its 'trainer ended'" % (T2_ENDED, T2_ENDED + 60)),
    ("owner2", [dict(event="trainer started", created=T2_CREATED + 30), dict(event="trainer ended", at_s=T2_ENDED)],
     "records [%r] for its 'trainer started' event(s)" % (T2_CREATED + 30)),
    ("owner_log", [dict(event="trainer started", created=R.T_CREATED)], "records none for its 'trainer ended' event(s)"),
], ids=["recovery-no-end", "recovery-two-ends", "recovery-another-start", "adoption-no-end"])
def test_the_wall_cap_is_read_from_both_run_owners_receipts(world, log, events, needle):
    def damage(w):
        w[log].write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    _refused_everywhere(world, needle, damage)


def test_a_missing_run_owner_log_is_refused(world):
    w = world()
    w["owner2"].unlink()
    _refused_written(w, "the run owner's log %s cannot be read" % w["owner2"])


def test_a_route_wall_cap_that_is_not_the_remaining_time_is_refused(world):
    """A route whose pinned cap is not floor(original - both lifetimes): the arithmetic itself is checked."""
    w = world(wall_cap_s=WALL_L + 1)
    _refused_written(w, "the link route's wall cap %d s is not the adoption's %d s less both trainers' 100.500 + 299.750 "
                        "s" % (WALL_L + 1, R.WALL_A), wall_cap_s=WALL_L + 1)


def test_a_start_step_other_than_the_stop_is_refused(world):
    _refused_written(world(), "starts at model step 1600; the link's start_step reads 1599", start_step=LINK_STOP - 1,
                     max_model_steps=R.END - LINK_STOP + 1)


def test_the_duration_is_bounded(world):
    _refused_written(world(), "the link's duration is not bounded: start_step 1600, end_step 1600, max_model_steps 0",
                     end_step=LINK_STOP, max_model_steps=0)


def test_a_trainer_lifetime_that_is_not_above_zero_is_refused(world):
    """A route whose first trainer's receipted end is before its start: no remaining time is computed from it."""
    w = world()
    _pin(w, trainers=((str(w["owner_log"]), R.T_ENDED, R.T_CREATED), (str(w["owner2"]), T2_CREATED, T2_ENDED)))
    _refused_written(w, "the link route's trainer lifetimes [%r, %r] are not each above 0 s" % (
        R.T_CREATED - R.T_ENDED, T2_ENDED - T2_CREATED))


def test_the_stop_fields_are_read_in_the_start_runs_stats(world):
    """The stopped recovery run's preserved stats without the declared stop lines' field: the watcher would never judge
    them (the stats are not among the MANIFEST's hashed files, so only this check sees it)."""
    w = world()
    (w["run3"] / "replay_training_stats.jsonl").write_text("".join(json.dumps(dict(type="mimo", step=s, elapsed_s=e))
                                                                   + "\n" for s, e in STATS_L), encoding="utf-8")
    _refused_written(w, "its field causal_pe_per_modality.vision reads no number in the newest full stats record of the "
                        "start checkpoint's run")


# ------------------------------------------------------------------------------ (7) THE GATES
def test_an_inherited_exception_is_carried_unchanged(world):
    w = world()
    for field in TP.ADOPTION_EXCEPTION_FIELDS:
        exc = _exceptions_l(w)
        exc["4"] = dict(exc["4"], **{field: ["another"] if field == "evidence" else "rewritten"})
        _refused_written(w, "the link's exception for gate 4 Local stability changes the used recovery's %s" % field,
                         exceptions=exc)


@pytest.mark.parametrize("field", TP.RECOVERY_CARRIED_FIELDS)
def test_each_exception_writes_where_it_comes_from_and_its_applicability(world, field):
    w = world()
    exc = _exceptions_l(w)
    exc["2"] = {k: v for k, v in exc["2"].items() if k != field}
    _refused_written(w, "the link's exception for gate 2 Gradient flow lacks its %s" % field.replace("_", " "),
                     exceptions=exc)


def test_an_exception_never_reads_a_status_other_than_fail(world):
    w = world()
    exc = _exceptions_l(w)
    exc["5"] = dict(exc["5"], status="PASS")
    _refused_written(w, "the link's exception for gate 5 reads status PASS", exceptions=exc)


def test_a_gate_the_used_recovery_has_no_exception_for_is_refused(world):
    """The recovery file edited (its exception for gate 3 taken out) and its new sha256 pinned, so only the missing
    exception differs."""
    w = world()
    b = _lblock(w)
    held = json.loads(TP.recovery_path(w["fp"], w["root"]).read_text(encoding="utf-8"))
    held["block"]["exceptions"].pop("3")
    TP._save(TP.recovery_path(w["fp"], w["root"]), held)
    sha = _fsha(TP.recovery_path(w["fp"], w["root"]))
    _pin(w, recovery_sha256=sha)
    b["predecessor"] = dict(b["predecessor"], recovery_sha256=sha)
    with pytest.raises(TP.PreflightError) as exc:
        TP.set_link(w["fp"], b, "tester", root=w["root"], registry=w["reg"], src=w["src"])
    assert "gate 3 Scale/conditioning: the used recovery names no exception for it, so none is carried" in str(
        exc.value), str(exc.value)


def test_a_gate_without_its_exception_is_refused(world):
    w = world()
    exc = {n: x for n, x in _exceptions_l(w).items() if n != "6"}
    _refused_written(w, "gate 6 Objective/test alignment reads FAIL and the link names no complete exception",
                     exceptions=exc)


def test_a_gate_judged_PASS_is_refused_even_with_the_record_pinned(world):
    w = world()
    TP.set_gate(w["fp"], 7, "PASS", "claimed without a run", ["any text"], "tester", root=w["root"])
    sha = _fsha(TP.record_path(w["fp"], w["root"]))
    _pin(w, record_sha256=sha)
    _refused_written(w, "gate 7 End-to-end causal dry run reads PASS; a link launches only with every gate FAIL (the "
                        "used adoption's decision, which the link keeps")


def test_the_exceptions_form(world):
    w = world()
    exc = dict(_exceptions_l(w), x=dict(_exceptions_l(w)["1"]))
    _refused_written(w, "the link's exception 'x' is not a gate number", exceptions=exc)
    _refused_written(w, "the link's exception 9 is not one of the seven gates",
                     exceptions=dict(_exceptions_l(w), **{"9": _exceptions_l(w)["1"]}))
    _refused_written(w, "the link's exception for gate 5 must be", exceptions=dict(_exceptions_l(w), **{"5": "x"}))
    _refused_written(w, "the link's exceptions must be a block", exceptions=[["1", "x"]])


# ------------------------------------------------------------------------------ (8) THE CARRIED CLOCK AND THE STOP POLICY
def test_the_carried_clock_lets_the_short_run_judge_its_quarter_hour_line(world):
    """On the trainer's own clock the line from run-hour 0.25 is first judged after the link's 400 model steps; on the
    carried clock (2,240 s carried, more than its 900 s) as early as the line judged from run-hour 0. link_first_readings
    with nothing carried is first_readings."""
    w = world()
    b = _lblock(w)
    own = TP.link_first_readings(b, 0.0, 100)
    assert own == TP.first_readings(b, 100)
    assert own["vision_causal_error"] < 400 < own["vision_from_a_quarter_hour"], own
    carried = TP.link_first_readings(b, CARRIED_S, 100)
    assert carried == {"vision_causal_error": own["vision_causal_error"],
                       "vision_from_a_quarter_hour": own["vision_causal_error"]}, carried
    assert TP.link_first_reading_problems(b, CARRIED_S, 100, 400) == []
    late = TP.link_first_reading_problems(b, 0.0, 100, 400)
    assert len(late) == 1 and "before the watcher first judges stop line vision_from_a_quarter_hour" in late[0], late


def test_with_no_carried_clock_a_line_judged_after_the_end_is_refused(world):
    w = world(carried_runs=(), carried_s=0.0)
    _refused_written(w, "the link ends 400 model steps after its start, before the watcher first judges stop line "
                        "vision_from_a_quarter_hour", carried_clock=dict(runs=[], seconds=0.0, basis="nothing carried"))
    w2 = world("launch")
    _write(w2)
    _pin(w2, carried_runs=(), carried_s=0.0)
    _edit_link(w2, carried_clock=dict(runs=[], seconds=0.0, basis="nothing carried"))
    why = _refusal(w2)
    assert why and "before the watcher first judges stop line vision_from_a_quarter_hour" in why, why


@pytest.mark.parametrize("clock, needle", [
    (lambda w: dict(runs=[str(w["run2"]), str(w["run3"])], seconds=CARRIED_S, basis="b"),
     "pins the hourly watcher's carried runs"),
    (lambda w: dict(runs=[str(w["run3"]), str(w["run2"]), str(w["p1"] / "run")], seconds=CARRIED_S, basis="b"),
     "pins the hourly watcher's carried runs"),
    (lambda w: dict(runs=[str(w["p1"] / "run"), str(w["run2"]), str(w["run3"])], seconds=CARRIED_S + 3600, basis="b"),
     "pins the carried clock %r s; the link's carried_clock reads %r" % (CARRIED_S, CARRIED_S + 3600)),
], ids=["a-run-left-out", "another-order", "more-seconds"])
def test_the_carried_clock_is_the_pinned_one(world, clock, needle):
    w = world()
    _refused_written(w, needle, carried_clock=clock(w))


def test_the_pinned_carried_seconds_are_the_preserved_runs_sum(world):
    w = world(carried_s=CARRIED_S + 3600)
    _refused_written(w, "the carried runs' newest full stats records read %r training seconds in all" % CARRIED_S,
                     carried_clock=dict(runs=[str(w["p1"] / "run"), str(w["run2"]), str(w["run3"])],
                                        seconds=CARRIED_S + 3600, basis="b"))


def test_a_carried_run_without_records_or_outside_the_preservations_is_refused(world):
    w = world()
    (w["p1"] / "run" / "replay_training_stats.jsonl").write_text("", encoding="utf-8")
    _refused_written(w, "the carried run %s holds no full stats record with its training seconds" % (w["p1"] / "run"))
    w2 = world("outside")
    outside = w2["a_run"]
    _pin(w2, carried_runs=(str(outside),), carried_s=600.0)
    _refused_written(w2, "the carried run %s is not a preserved copy" % outside,
                     carried_clock=dict(runs=[str(outside)], seconds=600.0, basis="b"))


@pytest.mark.parametrize("clock, needle", [
    ("x", "the link's carried_clock must be a block"),
    (dict(runs="one", seconds=1.0, basis="b"), "the link's carried_clock runs must be a list"),
    (dict(runs=[], seconds=-1.0, basis="b"), "the link's carried_clock seconds must be a number of seconds, 0 or more"),
    (dict(runs=[], seconds=1.0), "the link's carried_clock lacks its basis"),
], ids=["not-a-block", "runs-text", "negative", "no-basis"])
def test_the_carried_clocks_form(world, clock, needle):
    _refused_written(world(), needle, carried_clock=clock)


@pytest.mark.parametrize("field, value", [("rule", "M4"), ("sense", "vision"), ("crossing", "any"), ("mode", "stop"),
                                          ("watch", "docs/another_watch.py"), ("decision_words", "M6 is an alert")],
                         ids=["rule", "sense", "crossing", "mode", "watch", "decision-words"])
def test_the_stop_policy_exception_is_the_pinned_one(world, field, value):
    w = world()
    pol = dict(_lblock(w)["stop_policy_exception"], **{field: value})
    _refused_written(w, "pins the stop-policy exception's %s" % field.replace("_", " "), stop_policy_exception=pol)


@pytest.mark.parametrize("field", ["still_stops", "applicability"])
def test_the_stop_policy_exception_says_what_still_stops_and_where_it_applies(world, field):
    w = world()
    pol = {k: v for k, v in _lblock(w)["stop_policy_exception"].items() if k != field}
    _refused_written(w, "the link's stop_policy_exception lacks its %s" % field.replace("_", " "),
                     stop_policy_exception=pol)


def test_the_stop_policy_exception_is_one_scoped_exception(world):
    w = world()
    pol = dict(_lblock(w)["stop_policy_exception"], senses="all")
    _refused_written(w, "the link's stop_policy_exception names senses beyond its fields", stop_policy_exception=pol)
    _refused_written(w, "the link's stop_policy_exception must be a block", stop_policy_exception=["M6"])


def test_the_stop_policy_words_are_the_decisions(world):
    """A route whose policy words are not in the decision message's text: refused (the block names the same words)."""
    w = world(stop_policy=dict(rule="M6", watch=WATCH, sense="proprioception", crossing="finite", mode="alert",
                               decision_words="words no decision wrote"))
    pol = dict(_lblock(w)["stop_policy_exception"], decision_words="words no decision wrote")
    _refused_written(w, "the link route's stop-policy words are not in the decision message", stop_policy_exception=pol)


# ------------------------------------------------------------------------------ (9) THE ROWS, THE LAUNCH, THE FORM
@pytest.mark.parametrize("row, needle", [
    ("recover_done_row", "the link's completion_jobs_row is the used recovery's row recover_done_row"),
    ("adopt_done_row", "the link's completion_jobs_row is the used adoption's row adopt_done_row"),
    ("recover_stop_row", "the link's completion_jobs_row is the used recovery's row recover_stop_row"),
], ids=["recovery-completion", "adoption-completion", "recovery-training"])
def test_the_completion_row_is_never_an_earlier_blocks(world, row, needle):
    w = world(completion_row=row)
    _refused_written(w, needle, completion_jobs_row=row)


def test_the_training_row_is_never_an_earlier_blocks(world):
    w = world(run_row="recover_stop_row")
    _refused_written(w, "the link's stop_lines_row is the used recovery's row recover_stop_row",
                     stop_lines_row="recover_stop_row")


@pytest.mark.parametrize("row", ["link_stop_row", "link_done_row"])
def test_the_links_rows_must_read_written_or_queued(world, row):
    w = world()
    R._edit_rows(w["reg"], row, status="done")
    _refused_written(w, "reads status done; a link's row")


def test_a_failed_attempt_refuses_a_second_launch(world):
    w = world()
    _write(w)
    _note(w, _folder(w, "run_failed"))
    assert TP.link_available(_held(w)["block"])                         # the link file: no use
    retry = TP.apply_sets(w["cmd_l"], [("--latent-body-output", str(w["tmp"] / "lat_retry"))])
    why = _refusal(w, retry)
    assert why and ATTEMPT in why, why
    with pytest.raises(TP.PreflightError, match=ATTEMPT):
        _note(w, _folder(w, "run_retry"))


def test_a_row_recording_another_link_is_refused(world):
    w = world()
    R._edit_rows(w["reg"], "link_stop_row", **{TP.LINK_ROW_FIELD: dict(fingerprint=OTHER_FP, attempts=[])})
    _refused_written(w, "the training row link_stop_row records the launch of another link")


@pytest.mark.parametrize("flag, where, needle", [
    ("--out", "r_run", "the used recovery's run folder"),
    ("--checkpoint-storage", "a_run", "the used adoption's run folder"),
    ("--latent-body-output", "r_lat", "the used recovery's latent body folder"),
    ("--latent-body-output", "a_lat", "the used adoption's latent body folder"),
], ids=["out-recovery-run", "storage-adoption-run", "latent-recovery", "latent-adoption"])
def test_the_launch_never_writes_into_an_earlier_run(world, flag, where, needle):
    w = world()
    inside = w[where] / "link_inside"
    if flag == "--latent-body-output":
        _write(w, run_flags={"--latent-body-resume": str(w["latent3"]), "--latent-body-output": str(inside)})
    else:
        _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd_l"], [(flag, str(inside))]))
    assert why and "the launch's %s %s lies in %s" % (flag, inside, needle) in why, why


@pytest.mark.parametrize("name", ["new_run_r", "adoption_run"], ids=["the-recovery-runs-name", "the-adoption-runs-name"])
def test_the_launchs_out_never_takes_an_earlier_runs_folder_name(world, name):
    """The trainer writes the worlds companions into <--caregiver-record-dir>/checkpoint_state/<--out's folder name>."""
    w = world()
    _write(w)
    out = w["tmp"] / "elsewhere" / name
    why = _refusal(w, TP.apply_sets(w["cmd_l"], [("--out", str(out)), ("--checkpoint-storage", str(out / "c"))]))
    assert why and "the launch's --out %s has the folder name of" % out in why and "checkpoint_state/%s" % name in why, why


@pytest.mark.parametrize("flag", ["--out", "--checkpoint-storage", "--latent-body-output"])
def test_the_launch_writes_new_folders_outside_the_preservations(world, flag):
    w = world()
    inside = w["tmp"] / "preserved" / "new_folder"
    if flag == "--latent-body-output":
        _write(w, run_flags={"--latent-body-resume": str(w["latent3"]), "--latent-body-output": str(inside)})
    else:
        _write(w)
    why = _refusal(w, TP.apply_sets(w["cmd_l"], [(flag, str(inside))]))
    assert why and "the launch's %s %s lies in the preserved folder" % (flag, inside) in why, why
    if flag != "--latent-body-output":
        why = _refusal(w, TP.apply_sets(w["cmd_l"], [(flag, str(w["run3"]))]))
        assert why and "exists already: a link writes a new folder" in why, why


def test_the_run_flags_the_link_names_are_carried_exactly(world):
    w = world()
    _write(w)
    for flag, value in (("--latent-body-output", str(w["l_lat"]) + "_other"),
                        ("--latent-body-resume", str(w["latent3"]) + ".other")):
        why = _refusal(w, TP.apply_sets(w["cmd_l"], [(flag, value)]))
        assert why and "the launch's %s reads %s" % (flag, value) in why and "(run_flags, compared exactly)" in why, why


def test_the_run_flags_name_both_latent_body_flags_and_a_new_folder(world):
    w = world()
    _refused_written(w, "the link's run_flags name --latent-body-output",
                     run_flags={"--latent-body-resume": str(w["latent3"])})
    w["l_lat"].mkdir()
    _refused_written(w, "exists already; a link writes a new folder")


@pytest.mark.parametrize("key", TP.LINK_FIELDS)
def test_every_field_is_required(world, key):
    w = world()
    b = _lblock(w)
    b.pop(key)
    with pytest.raises(TP.PreflightError, match="the link's %s is missing" % key.replace("_", " ")):
        TP.set_link(w["fp"], b, "tester", root=w["root"], registry=w["reg"], src=w["src"])


@pytest.mark.parametrize("change, needle", [
    (lambda w: dict(predecessor=_lpred(w, stopped_step="1600")), "predecessor stopped_step must be a whole number"),
    (lambda w: dict(predecessor=_lpred(w, recovery_sha256="ABC")), "the link's predecessor recovery_sha256 must be a "
                                                                   "sha256"),
    (lambda w: dict(predecessor=_lpred(w, recovery_run=None)), "the link's predecessor recovery_run must be text"),
    (lambda w: dict(predecessor=_lpred(w, adoption=None)), "the link's predecessor adoption must be text"),
    (lambda w: dict(code=dict(commit=R.TO_COMMIT)), "the link's code code_hash must be text"),
    (lambda w: dict(code=["a"]), "the link's code must be a block"),
    (lambda w: dict(role="recovery"), "the link's role reads recovery"),
    (lambda w: dict(repeats=2), "a link is used once"),
    (lambda w: dict(target_gates=[5]), "a link targets no gate"),
    (lambda w: dict(start_step="1600"), "start_step must be a whole number"),
    (lambda w: dict(authorization=["words"]), "the link's authorization must be text"),
], ids=["stopped-step", "recovery-sha", "recovery-run", "adoption-file", "code-hash", "code-block", "role", "repeats",
        "targets", "start-str", "authorization-text"])
def test_the_blocks_form(world, change, needle):
    w = world()
    _refused_written(w, needle, **change(w))


def test_the_noting_of_a_link_needs_src(world):
    w = world()
    _write(w)
    run = _folder(w, "run_a")
    for fn in (_note, _mark):
        with pytest.raises(TP.PreflightError, match="a link's attempt and launch are noted with src"):
            fn(w, run, src=False)


def test_set_link_needs_a_record_and_who_writes_it(world):
    w = world()
    with pytest.raises(TP.PreflightError, match="no record for configuration"):
        TP.set_link(OTHER_FP, _lblock(w), "tester", root=w["root"], registry=w["reg"], src=w["src"])
    with pytest.raises(TP.PreflightError, match="who writes it"):
        TP.set_link(w["fp"], _lblock(w), " ", root=w["root"], registry=w["reg"], src=w["src"])


def test_the_stop_criteria_are_checked_as_a_continuations(world):
    w = world()
    _refused_written(w, "no stop criterion has mode stop", stop_criteria=[dict(c, mode="alert") for c in STOP_CRITERIA_L])


def test_the_command_line_writes_a_link_and_marks_its_launch(world, monkeypatch, capsys):
    w = world()
    path = w["tmp"] / "link_block.json"
    path.write_text(json.dumps(_lblock(w)), encoding="utf-8-sig")
    set0 = TP.set_link
    monkeypatch.setattr(TP, "set_link", lambda f, b, by, *x, **k: set0(f, b, by, root=w["root"], **k))
    assert TP.main(["link", w["fp"], "--block-file", str(path), "--by", "tester"]) == 2
    assert "link needs --src" in capsys.readouterr().out
    assert TP.main(["link", w["fp"], "--src", str(w["src"]), "--by", "tester"]) == 2
    assert "link needs --block-file" in capsys.readouterr().out
    assert TP.main(["link", w["fp"], "--block-file", str(path), "--src", str(w["src"]), "--registry",
                    str(w["reg"]), "--by", "tester"]) == 0
    assert "link written" in capsys.readouterr().out
    mark0 = TP.mark_launched
    monkeypatch.setattr(TP, "mark_launched", lambda f, r, by, *x, **k: mark0(f, r, by, root=w["root"], **k))
    run = _folder(w, "run_cli")
    assert TP.main(["launched", w["fp"], "--run", str(run), "--link", "--recovery", "--by", "t"]) == 2
    assert "launched marks one block" in capsys.readouterr().out
    assert TP.main(["launched", w["fp"], "--link", "--by", "t"]) == 2
    assert "launched needs --run, the link's folder" in capsys.readouterr().out
    assert TP.main(["launched", w["fp"], "--run", str(run), "--link", "--registry", str(w["reg"]), "--by",
                    "tester"]) == 2
    assert "noted with src" in capsys.readouterr().out
    assert TP.main(["launched", w["fp"], "--run", str(run), "--link", "--src", str(w["src"]), "--registry",
                    str(w["reg"]), "--by", "tester"]) == 0
    assert "link marked launched" in capsys.readouterr().out
    assert _held(w)["block"]["run"] == str(run.resolve())
    assert _rows(w)["link_stop_row"][TP.LINK_ROW_FIELD]["launched_at"]


def test_the_command_lines_check_reads_another_checkout_registry_and_folder(world, monkeypatch, capsys):
    w = world()
    _write(w)
    path = w["tmp"] / "cmd_l.json"
    path.write_text(json.dumps(w["cmd_l"]), encoding="utf-8")
    assert TP.main(["check", "--cmd-file", str(path), "--src", str(w["src"]), "--registry", str(w["reg"]), "--root",
                    str(w["root"])]) == 0
    assert "ALLOWED: configuration %s may launch." % w["fp"] in capsys.readouterr().out
    assert TP.main(["check", "--cmd-file", str(path), "--src", str(w["src"]), "--registry", str(w["reg"])]) == 1
    assert "has no preflight record" in capsys.readouterr().out          # production's folder by default: not this one
