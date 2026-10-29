# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE LATENT BODY SHADOW'S DECLARED STARTING STATE, AND THE CHECK OF A RUN'S SHADOW RECORDS (Oct 10, 2026).

Defect register entry LATENT_SHADOW_REJECTS_REPAIRED_FINGER_CONTRACT; design note
docs/gain_of_function_v4/latent_contract_repair/LATENT_CONTRACT_REPAIR.md. This file is a tool: it lies under
mimo_bridge/tools/, outside the trainer code (lem/trainer_code.py SKIP), and the trainer never imports it.

WHY IT EXISTS. With --finger-joint-servo on the trainer writes each completed transition's command contract as
wrist_relative_joint_v1+finger_joint_servo_v1 (train_mimo.py joins limb_target_control.action_contract_parts to the hand's
wrist_relative_joint_v1). The shadow's configuration of Sep 30 declares wrist_relative_joint_v1, and the trajectory
extractor refuses a transition whose contract differs from the configuration's (lem/latent_body/trajectory.py,
TrajectoryExtractor.observe), so on Oct 10 the shadow stopped at its first transition and learned nothing in the finger
arm. The repair declares the repaired contract in its own configuration file
(docs/gain_of_function_v4/latent_body_shadow_config_h84_finger_joint_servo_v1_2026-10-10.json: the Sep 30 file with
action_contract and measurement changed). A saved shadow state is bound to its configuration by the SHA-256 of the
configuration (lem/latent_body/checkpoint.py validate_header), so a state learned under one contract never loads under
the other. This tool makes the starting state of a run under a declared configuration an explicit, named file.

THE TWO STARTS (subcommands fresh and copy). Both write a state with 0 updates and 0 examples, an empty extractor history
(no observation of another run), Adam's running gradient averages empty, and the configuration's own private generator;
they differ in the weights only:
  fresh  the weights a ShadowLearner builds from the configuration's seed (the same numbers a shadow builds when its
         output folder holds no state).
  copy   the weights of a named saved state recorded under ANOTHER contract, loaded first under that state's own
         recorded configuration (its header and every number checked as a resume checks them), then copied into the
         new configuration's model (strict: every tensor, same shapes). The two configurations may differ only in
         action_contract and measurement (CARRIED_KEYS); the source file is read, never written.
The file holds the learner's state (lem/latent_body/learning.py ShadowLearner.state_dict) and a "start" record of
primitives (kind, configuration, source); a sidecar <file>.json holds the same record with the file's SHA-256. The
shadow's runtime reads the state through checkpoint.load and ignores the "start" record. A start is never written over an
existing file, under a Saved folder, into a preserved shadow folder (PRESERVED) or beside a shadow's records.

THE CHECK OF A RUN (subcommand verify; the registry row draft latent_contract_check_481800): reads a run's shadow folder,
its arm log and args.json and answers four things, each a reading with its pass line:
  identity    the trainer ran --finger-joint-servo on; the shadow's started record holds the declared configuration and
              the start file; every trajectory row carries the declared contract (a row exists only after the runtime
              compared the transition's contract with the frame's and the extractor compared it with the configuration);
              no failed record and no "Latent body shadow failed" line.
  updates     the lesson records' update counts run 1, 2, ..., n with no gap, n >= 2 and n >= (model steps - 2) - 1
              (both control arms of Oct 10 recorded 998 updates in 1,000 model steps: the first example needs two
              earlier observations of the body).
  rows        the trajectory rows number the last lesson's example count, each labelled with the declared contract.
  restore     latest.pt loads under the declared configuration with the last lesson's update and example counts, is
              refused under the old configuration, learns once more from the run's own last rows (updates n + 1, finite),
              and a ShadowRuntime started with --latent-body-resume auto on a copy of it continues at n.
  boundary    the resume boundary (Oct 10, 2026, the first check's FAIL): a launch resumed from a checkpoint saved under
              the pose servo closes, at its first model step, the open transitions the episode companion restores; they
              were executed before the checkpoint under wrist_relative_joint_v1. Under the runtime of the branch's commit
              marked NOT FOR MERGE the shadow records them in one excluded_transitions record and learns on; the answer
              passes when that record is the only one, at the shadow's first model step, and holds only transitions
              executed before it under another contract (and, when given, the expected number).
Optionally the old contract's state (a preserved latest.pt) is loaded read only under its own configuration.

    python -m mimo_bridge.tools.latent_body_start_state fresh  --config CFG --out FILE [--levels L] [--layout trunk_v1]
    python -m mimo_bridge.tools.latent_body_start_state copy   --config CFG --from STATE --out FILE [--levels L] [--layout]
    python -m mimo_bridge.tools.latent_body_start_state show   --config CFG --state FILE [--levels L] [--layout]
    python -m mimo_bridge.tools.latent_body_start_state verify --config CFG --start-record FILE.json --shadow DIR
               --steps N [--log FILE] [--args FILE] [--old-config CFG --old-state FILE] --out RESULT.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

REPAIRED_CONTRACT = "wrist_relative_joint_v1+finger_joint_servo_v1"
OLD_CONTRACT = "wrist_relative_joint_v1"
# the only configuration keys a copied start may differ in from its source's configuration
CARRIED_KEYS = ("action_contract", "measurement")
# saved shadow folders this tool never writes into (LATENT_SHADOW_STATE_MIXED_BY_ARMS: the mixed state and its copy)
PRESERVED = tuple(Path(p) for p in ("D:/UEMA_latent_body/seg41", "D:/UEMA_latent_body/seg41_v2",
                                    "D:/UEMA_latent_body/seg41_d3"))
# a shadow's own records: a start is never written beside them
SHADOW_RECORDS = ("events.jsonl", "trajectory.jsonl", "plans.jsonl")
DROPPED_BY_COPY = ("Adam's running gradient averages and step counts", "the extractor's saved observations",
                   "the update and example counts", "the private random generator's state")
FAILED_LINE = "Latent body shadow failed"
TOOL_VERSION = "latent_body_start_state_v1"


def clock() -> str:
    from mimo_bridge.tools.test_registry import clock as registry_clock
    return registry_clock()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(path) -> str:
    """SHA-256 of a text file with every CRLF read as LF (lem/trainer_code.file_hash, the hash the training preflight
    gives a file a launch flag names), so two checkouts of one commit agree."""
    from lem.trainer_code import file_hash
    return file_hash(path)


def load_config(path) -> dict:
    from lem.latent_body.learning import validate_config
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def tree_for(levels="arm,hand,index", layout="trunk_v1"):
    from lem.latent_body.tree import body_tree
    from mimo_bridge.body_layout import names_for
    return body_tree(names_for(layout), levels)


def _under(path: Path, folder: Path) -> bool:
    p, f = str(Path(path).resolve()).replace("\\", "/").lower(), str(Path(folder).resolve()).replace("\\", "/").lower()
    return p == f or p.startswith(f.rstrip("/") + "/")


def target_problems(out) -> list:
    """Why a start may not be written at `out` (empty: it may)."""
    out = Path(out)
    problems = []
    if out.exists():
        problems.append("%s exists: a start is never written over a file" % out)
    if "/saved/" in str(out.resolve()).replace("\\", "/").lower() + "/":
        problems.append("%s lies under a Saved folder" % out)
    for folder in PRESERVED:
        if _under(out, folder):
            problems.append("%s lies in the preserved shadow folder %s" % (out, folder))
    beside = [n for n in SHADOW_RECORDS if (out.parent / n).exists()]
    if beside:
        problems.append("%s holds a shadow's records (%s): a start is written into a folder of its own"
                        % (out.parent, ", ".join(beside)))
    return problems


def fresh_start(tree, config):
    from lem.latent_body.learning import ShadowLearner
    return ShadowLearner(tree, config)


def config_differences(a: dict, b: dict) -> list:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def copy_start(tree, config, source_state: dict):
    """(learner under `config` holding the source's weights only, the source's summary). The source is loaded under its
    own recorded configuration first, so its header, shapes and every number are checked as a resume checks them."""
    from lem.latent_body.checkpoint import fingerprint
    from lem.latent_body.learning import ShadowLearner
    source_config = source_state.get("config")
    if not isinstance(source_config, dict):
        raise ValueError("the source state records no configuration")
    differ = config_differences(source_config, config)
    extra = [k for k in differ if k not in CARRIED_KEYS]
    if extra:
        raise ValueError("the source's configuration differs in %s, beyond %s: its weights mean other numbers"
                         % (extra, list(CARRIED_KEYS)))
    if source_config.get("action_contract") == config.get("action_contract"):
        raise ValueError("the source already holds the contract %s: resume it under its own configuration, do not copy"
                         % config.get("action_contract"))
    source = ShadowLearner(tree, source_config)
    source.load_state_dict(source_state)
    learner = ShadowLearner(tree, config)
    learner.model.load_state_dict(source.model.state_dict(), strict=True)
    summary = dict(action_contract=source_config.get("action_contract"), configuration=fingerprint(source_config),
                   updates=int(source.updates), examples=int(source.examples), carried="model weights",
                   dropped=list(DROPPED_BY_COPY))
    return learner, summary


def write_start(out, learner, start: dict) -> dict:
    """The learner's state with its start record, written atomically at `out`, read back under the same configuration
    (0 updates, the same weights), and the sidecar <out>.json. Returns the sidecar's record."""
    import torch
    from lem.latent_body.learning import ShadowLearner
    from mimo_bridge import copied_execution as CE
    out = Path(out)
    problems = target_problems(out)
    if problems:
        raise SystemExit("refused: " + "; ".join(problems))
    out.parent.mkdir(parents=True, exist_ok=True)
    state = learner.state_dict()
    state["start"] = json.loads(json.dumps(start))
    descriptor, temporary = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    back = torch.load(out, map_location="cpu", weights_only=True)
    check = ShadowLearner(learner.tree, learner.config)
    check.load_state_dict(back)
    if check.updates != 0 or check.examples != 0 or not CE.equal_tree(check.model.state_dict(),
                                                                      learner.model.state_dict()):
        raise RuntimeError("the written start does not read back as written")
    record = dict(start, path=str(out), sha256=sha256_file(out), bytes=out.stat().st_size)
    Path(str(out) + ".json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")
    return record


def start_record(kind, config_path, config, tree, levels, layout, source=None) -> dict:
    from lem.latent_body.checkpoint import fingerprint
    return dict(tool=TOOL_VERSION, tool_sha256=sha256_file(__file__), kind=kind, written=clock(),
                config_path=str(config_path), config_sha256=text_sha256(config_path),
                configuration=fingerprint(config), action_contract=config["action_contract"], tree=tree.fingerprint,
                levels=levels, layout=layout, updates=0, examples=0, source=source)


def state_summary(path, tree, config) -> dict:
    """{action_contract, configuration, updates, examples, start} of a saved state loaded under `config`."""
    import torch
    from lem.latent_body.checkpoint import fingerprint
    from lem.latent_body.learning import ShadowLearner
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    learner = ShadowLearner(tree, config)
    learner.load_state_dict(state)
    return dict(path=str(path), action_contract=state["config"]["action_contract"],
                configuration=fingerprint(learner.config), updates=learner.updates, examples=learner.examples,
                start=state.get("start"))


# ============================================================================================== the check of a run
def read_jsonl(path) -> list:
    out = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                out.append(json.loads(line))
    return out


def example_from_row(row):
    from lem.latent_body.trajectory import Example, Observation, tensor
    return Example(Observation.from_dict(row["history"]), Observation.from_dict(row["start"]),
                   Observation.from_dict(row["end"]), tensor(row["executed"]), row["action_contract"])


def runtime_resume_updates(config_path, latest, handles, levels) -> int:
    """The update count a ShadowRuntime reads at its start with --latent-body-resume auto, on a copy of `latest` in a
    temporary output folder (the trainer's own resume path, latent_body_runtime.ShadowRuntime.__init__)."""
    from types import SimpleNamespace
    from mimo_bridge.latent_body_runtime import ShadowRuntime
    with tempfile.TemporaryDirectory() as folder:
        shutil.copyfile(latest, Path(folder) / "latest.pt")
        args = SimpleNamespace(latent_body_output=folder, latent_body_levels=levels,
                               latent_body_config=str(config_path), latent_body_resume="auto", latent_body_log_mb=0.0,
                               latent_body_save_every=1, latent_planner="off", eye_row_convention="legacy",
                               report_every=100)
        return int(ShadowRuntime(args, handles).learner.updates)


def reading(name, value, unit, meaning, baseline, ok) -> dict:
    return dict(name=name, value=value, unit=unit, meaning=meaning, baseline=baseline, statistically_clear="not judged",
                statistical_basis="", gof_material="GOF threshold not yet defined", passed=bool(ok))


def boundary_answer(excluded, contract, expected=None) -> tuple:
    """(ok, count, meaning) for the run's excluded_transitions records (the resume boundary, written by
    latent_body_runtime.ShadowRuntime.observe on the branch's commit marked NOT FOR MERGE): at most one record, at the
    shadow's first observed model step (its step equals its first_step), and every transition in it executed before
    that step (frame_t < first_step) under another contract than the declared one. expected: the number of excluded
    transitions the run must hold (None: not judged)."""
    transitions = [tr for e in excluded for tr in (e.get("transitions") or [])]
    ok = (len(excluded) <= 1
          and all(e.get("step") == e.get("first_step") and e.get("declared_contract") == contract for e in excluded)
          and all(int(tr.get("frame_t", 1 << 62)) < int(e.get("first_step", -1)) and tr.get("action_contract") != contract
                  for e in excluded for tr in (e.get("transitions") or []))
          and (expected is None or len(transitions) == int(expected)))
    meaning = ("%d transitions excluded in %d records%s; their contracts %s, executed at %s, the shadow's first model "
               "step %s" % (len(transitions), len(excluded),
                            "" if expected is None else " (expected %d)" % int(expected),
                            sorted({str(tr.get("action_contract")) for tr in transitions}),
                            sorted({tr.get("frame_t") for tr in transitions}),
                            sorted({e.get("first_step") for e in excluded})))
    return ok, len(transitions), meaning


def verify_run(shadow, config_path, start, steps, *, log=None, args_path=None, old_config_path=None, old_state=None,
               levels="arm,hand,index", layout="trunk_v1", handles=None, expected_excluded=None) -> dict:
    """The four answers of the module text for one run's shadow folder `shadow` (read only), and the resume boundary's
    excluded transitions (boundary_answer)."""
    import torch
    from lem.latent_body.learning import ShadowLearner
    shadow, config = Path(shadow), load_config(config_path)
    tree = tree_for(levels, layout)
    contract = config["action_contract"]
    problems, readings = [], []
    events = read_jsonl(shadow / "events.jsonl") if (shadow / "events.jsonl").is_file() else []
    started = [e for e in events if e.get("type") == "started"]
    failed = [e for e in events if e.get("type") == "failed"]
    lessons = [e for e in events if e.get("type") == "latent_body_lesson"]
    excluded = [e for e in events if e.get("type") == "excluded_transitions"]
    rows = read_jsonl(shadow / "trajectory.jsonl") if (shadow / "trajectory.jsonl").is_file() else []
    log_failed = None
    if log is not None:
        text = Path(log).read_text(encoding="utf-8", errors="replace")
        log_failed = text.count(FAILED_LINE)
    run_args = json.loads(Path(args_path).read_text(encoding="utf-8")) if args_path else None

    # identity
    servo = None if run_args is None else run_args.get("finger_joint_servo")
    labels = sorted({r.get("action_contract") for r in rows})
    # the started record names the declared start file (the explicit --latent-body-resume of the check's command)
    start_ok = (len(started) == 1 and started[0].get("config") == config
                and (start is None or (bool(started[0].get("resume"))
                                       and Path(str(started[0]["resume"])) == Path(str(start.get("path"))))))
    identity = (contract == REPAIRED_CONTRACT and start_ok and not failed and not log_failed
                and (run_args is None or (servo == "on"
                                          and Path(run_args.get("latent_body_config", "")) == Path(config_path)))
                and bool(rows) and labels == [contract])
    readings.append(reading(
        "rows whose command contract equals the declared one", sum(r.get("action_contract") == contract for r in rows),
        "rows", "of %d trajectory rows the shadow wrote (0 = the shadow learned from no transition, all = every "
        "transition the trainer wrote carried the declared contract %s); finger servo setting %r; started records %d "
        "holding the declared configuration %s; failed records %d; failure lines in the log %s"
        % (len(rows), contract, servo, len(started), start_ok, len(failed), log_failed),
        "Oct 10, 12:58:01 AM PDT: the finger arm under the Sep 30 configuration wrote 0 rows and one failed record "
        "(Action contract changed)", identity))
    if not identity:
        problems.append("identity: rows %d labelled %s, started records %d (declared configuration and start: %s), "
                        "failed records %d, failure lines %s, finger servo %r"
                        % (len(rows), labels, len(started), start_ok, len(failed), log_failed, servo))

    # the resume boundary: transitions executed before the shadow's first model step under another contract
    boundary_ok, n_excluded, boundary_meaning = boundary_answer(excluded, contract, expected_excluded)
    readings.append(reading(
        "transitions excluded at the resume boundary", n_excluded, "transitions",
        boundary_meaning + " (0 = no boundary transition carried another contract; one per body = the open transitions "
        "the episode companion restores, executed before the checkpoint)",
        "the companion of 481,800 restores 8 open transitions executed at model step 481,799 under "
        "wrist_relative_joint_v1 (evidence/companion_481800_contract.json)", boundary_ok))
    if not boundary_ok:
        problems.append("boundary: %s" % boundary_meaning)

    # updates
    counts = [int(e.get("updates", -1)) for e in lessons]
    expected = int(steps) - 2
    consecutive = counts == list(range(1, len(counts) + 1))
    updates_ok = consecutive and len(counts) >= 2 and len(counts) >= expected - 1
    readings.append(reading(
        "shadow updates", counts[-1] if counts else 0, "updates",
        "lesson records %d, their update counts consecutive from 1: %s (0 = no learning; %d expected for %d model "
        "steps, one per model step from the third)" % (len(counts), consecutive, expected, int(steps)),
        "both control arms of Oct 10: 998 updates in 1,000 model steps, first at model step 481,802", updates_ok))
    if not updates_ok:
        problems.append("updates: %d lesson records, consecutive from 1: %s, against at least %d"
                        % (len(counts), consecutive, max(2, expected - 1)))

    # rows
    examples = int(lessons[-1].get("examples", -1)) if lessons else 0
    batch_sum = sum(int(e.get("batch_examples", 0)) for e in lessons)
    rows_ok = bool(rows) and len(rows) == examples == batch_sum and labels == [contract]
    readings.append(reading(
        "trajectory rows", len(rows), "rows",
        "against the last lesson's example count %d and the lessons' batch sizes summed %d; labels %s (one label = "
        "no window mixed two contracts)" % (examples, batch_sum, labels),
        "both control arms of Oct 10: 7,984 rows, 8 per update", rows_ok))
    if not rows_ok:
        problems.append("rows: %d rows, last example count %d, batch sum %d, labels %s"
                        % (len(rows), examples, batch_sum, labels))

    # restore
    latest = shadow / "latest.pt"
    restore = dict(exists=latest.is_file())
    if latest.is_file():
        try:
            state = torch.load(latest, map_location="cpu", weights_only=True)
            learner = ShadowLearner(tree, config)
            learner.load_state_dict(state)
            restore.update(updates=learner.updates, examples=learner.examples,
                           contract=state["config"]["action_contract"])
            if old_config_path:
                try:
                    ShadowLearner(tree, load_config(old_config_path)).load_state_dict(state)
                    restore["refused_under_old"] = False
                except ValueError:
                    restore["refused_under_old"] = True
            last = [example_from_row(r) for r in rows if counts and r["end"]["frame"] == lessons[-1].get("step")]
            if last:
                metrics = learner.learn(last, train=True)
                restore["continued_to"] = int(metrics["updates"])
            restore["runtime_auto_updates"] = runtime_resume_updates(
                config_path, latest, list(handles or tree.handle_names), levels)
        except Exception as exc:          # recorded as the reading's problem, never raised
            restore["error"] = "%s: %s" % (type(exc).__name__, exc)
    n = counts[-1] if counts else None
    restore_ok = (restore.get("exists") and "error" not in restore and restore.get("updates") == n
                  and restore.get("examples") == examples and restore.get("contract") == contract
                  and restore.get("continued_to") == (None if n is None else n + 1)
                  and restore.get("runtime_auto_updates") == n
                  and (not old_config_path or restore.get("refused_under_old") is True))
    readings.append(reading(
        "saved shadow state restored", restore.get("updates"), "updates",
        "latest.pt read under the declared configuration: %s (the run's last lesson %s updates, %s examples; "
        "0 updates would mean the start, not the run)" % (json.dumps(restore, sort_keys=True), n, examples),
        "a state saved after every observation (--latent-body-save-every 1) holds the last lesson's counts", restore_ok))
    if not restore_ok:
        problems.append("restore: %s" % json.dumps(restore, sort_keys=True))

    # the old contract's state, read only
    old = None
    if old_state and old_config_path:
        try:
            old = state_summary(old_state, tree, load_config(old_config_path))
            old["loads"] = True
        except Exception as exc:
            old = dict(path=str(old_state), loads=False, error="%s: %s" % (type(exc).__name__, exc))
        readings.append(reading(
            "old contract's state loads", old.get("updates"), "updates",
            "%s read under %s: %s" % (old_state, old_config_path, json.dumps(old, sort_keys=True)),
            "LATENT_SHADOW_STATE_MIXED_BY_ARMS: 59,354 updates, the mixed state of Oct 6", old.get("loads")))
        if not old.get("loads"):
            problems.append("old state: %s" % old.get("error"))
    verdict = "pass" if not problems else "fail"
    return dict(tool=TOOL_VERSION, tool_sha256=sha256_file(__file__), written=clock(), shadow=str(shadow),
                config=str(config_path), config_sha256=text_sha256(config_path), contract=contract, steps=int(steps),
                start=start, verdict=verdict, problems=problems, readings=readings, restore=restore, old_state=old,
                counts=dict(lessons=len(lessons), rows=len(rows), started=len(started), failed=len(failed),
                            log_failed=log_failed, excluded=n_excluded))


# ============================================================================================== command line
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="sub", required=True)
    for name in ("fresh", "copy", "show", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--levels", default="arm,hand,index")
        p.add_argument("--layout", default="trunk_v1")
        if name in ("fresh", "copy"):
            p.add_argument("--out", required=True)
        if name == "copy":
            p.add_argument("--from", dest="source", required=True)
        if name == "show":
            p.add_argument("--state", required=True)
        if name == "verify":
            p.add_argument("--start-record", required=True)
            p.add_argument("--shadow", required=True)
            p.add_argument("--steps", type=int, required=True)
            p.add_argument("--log")
            p.add_argument("--args")
            p.add_argument("--old-config")
            p.add_argument("--old-state")
            p.add_argument("--expected-excluded", type=int, default=None,
                           help="transitions the resume boundary must exclude (default: not judged)")
            p.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    config = load_config(a.config)
    tree = tree_for(a.levels, a.layout)
    if a.sub == "fresh":
        record = write_start(a.out, fresh_start(tree, config),
                             start_record("fresh", a.config, config, tree, a.levels, a.layout))
        print(json.dumps(record, indent=1))
        return 0
    if a.sub == "copy":
        import torch
        source_path = Path(a.source)
        before = sha256_file(source_path)
        learner, summary = copy_start(tree, config, torch.load(source_path, map_location="cpu", weights_only=True))
        if sha256_file(source_path) != before:
            raise SystemExit("refused: the source changed while it was read")
        summary.update(path=str(source_path), sha256=before, bytes=source_path.stat().st_size,
                       modified=source_path.stat().st_mtime)
        record = write_start(a.out, learner, start_record("copy", a.config, config, tree, a.levels, a.layout,
                                                          source=summary))
        print(json.dumps(record, indent=1))
        return 0
    if a.sub == "show":
        print(json.dumps(state_summary(a.state, tree, config), indent=1, default=str))
        return 0
    start = json.loads(Path(a.start_record).read_text(encoding="utf-8"))
    result = verify_run(a.shadow, a.config, start, a.steps, log=a.log, args_path=a.args, old_config_path=a.old_config,
                        old_state=a.old_state, levels=a.levels, layout=a.layout,
                        expected_excluded=a.expected_excluded)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(dict(verdict=result["verdict"], problems=result["problems"]), indent=1))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
