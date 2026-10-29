# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The gain-of-function scoreboard: five stages, each stage's state computed from test registry rows.

The operator's recovery directive of Oct 5, 2026, section 13: "The top of the dashboard should answer one question:
What can the model reliably do now that it could not do before?" and "Internal repairs and tests are supporting work.
They do not count as gain of function."

THE FIVE STAGES and the evidence each requires are the directive's own table (STAGES below).

A STAGE'S STATE is computed on every read (scoreboard()), never typed:
  INPUT      the test registry's rows that name the stage as evidence: a row's field
                 "evidence_for": {"stage": one of the stage keys, "claim": one sentence of what the row shows}
             and the defect register's open GOF-BLOCKER defects whose "stage" is that stage.
  OPERATION  A row HOLDS as evidence when all of these are true: its outcome is PASS; its last run carries the
             runner's verified seal (test_registry.sealed: the PASS was written by a runner, not typed); its result is
             the one the runner sealed at the run's end (test_registry.runner_result: readings copied from the run's
             own result file; a reading typed into the row afterwards breaks the seal's digest) and carries one
             structured reading or more; and every reading says "yes" to both judgements, statistically clear and
             large enough to matter for gain of function.
             The stage is DEMONSTRATED when one row or more holds and no open GOF-BLOCKER defect names the stage.
             Else it is NOT DEMONSTRATED, with the reasons: no row names it; the rows that name it and why each does
             not hold; the open defects that block it.
  OUTPUT     per stage {key, name, required_evidence, state, reasons, evidence_rows, blocking_defects}; and the
             headline: gain of function is demonstrated only when the fifth stage, intentional function, is
             DEMONSTRATED. demonstrated_abilities lists the claims of the rows that hold for that stage.
"""
from __future__ import annotations

from pathlib import Path
import time

from . import test_registry as tr

STAGES = (
    ("SENSING", "Sensing works", "Relevant state is present in the sense representation."),
    ("PREDICTION", "Prediction works",
     "Prediction beats its proper baseline by a statistically clear and materially useful amount."),
    ("ACTION_CAUSALITY", "Action causality works",
     "Changing the command changes the predicted and real consequence where physics says it should."),
    ("PLANNER", "Planner works",
     "Planned command produces a better real physical outcome than the motor controller baseline."),
    ("INTENTIONAL_FUNCTION", "Intentional function",
     "Behavior is chosen, predicted, executed, repeated, generalized and retained."),
)
STAGE_KEYS = tuple(key for key, _, _ in STAGES)
FINAL_STAGE = "INTENTIONAL_FUNCTION"
STATES = ("DEMONSTRATED", "NOT DEMONSTRATED")


def evidence_stage(row):
    """The stage a row names as evidence (its evidence_for.stage when it is one of STAGE_KEYS), else None."""
    named = row.get("evidence_for") if isinstance(row, dict) else None
    stage = named.get("stage") if isinstance(named, dict) else None
    return stage if stage in STAGE_KEYS else None


def row_holds(row):
    """(True, None) when a row holds as evidence (the module text), else (False, why it does not)."""
    recorded = tr.recorded_outcome(row)           # the runner's record, never the row's copied "outcome" field
    if recorded != "PASS":
        return False, ("its outcome is %s" % recorded if recorded else
                       "it has no outcome written by a runner (status %s)" % row.get("status"))
    if not tr.sealed(row):
        return False, "its PASS does not carry the runner's verified seal"
    whole, current = tr.current_result(row)
    if not current:
        return False, "its result is the result of an earlier run, not of the run that passed"
    # READINGS COUNT ONLY FROM THE RUN'S OWN RESULT FILE, SEALED WITH THE RUN (the dashboard review of Oct 5, 2026,
    # finding F4): the runner copies the result file's readings into the result it writes and seals that result's
    # digest; readings typed into the row afterwards make the result differ from the sealed one
    result, why = tr.runner_result(row)
    if result is None:
        return False, "its result is not the one the runner sealed at the run's end (%s)" % why
    readings = [r for r in result.get("readings") or [] if isinstance(r, dict)]
    if not readings:
        return False, "its result carries no structured reading with the two judgements"
    for reading in readings:
        problems = tr.reading_problems(reading)
        if problems:
            return False, "its reading %s is malformed (%s)" % (reading.get("name"), "; ".join(problems))
        if reading.get("statistically_clear") != "yes":
            return False, "its reading %s is not statistically clear (%s)" % (reading.get("name"),
                                                                             reading.get("statistically_clear"))
        if reading.get("gof_material") != "yes":
            return False, "its reading %s is not judged large enough to matter for gain of function (%s)" % (
                reading.get("name"), reading.get("gof_material"))
    return True, None


def scoreboard(data_root=None, root=None, rows=None, defects=None, now=None) -> dict:
    """GET /api/scoreboard (the module text). `rows` and `defects` replace the files' for a test; `data_root` holds
    the test registry, `root` the defect register."""
    now = time.time() if now is None else float(now)
    out = dict(schema=1, read_at_text=tr.clock(now), states=list(STATES),
               question="What can the model reliably do now that it could not do before?")
    errors = []
    if rows is None:
        try:
            rows = tr.load(tr.registry_path(Path(data_root or tr.REPO)))["rows"]
        except tr.RegistryError as exc:
            rows = []
            errors.append("The test registry cannot be read (%s): no stage can be shown as demonstrated." % exc)
    if defects is None:
        try:
            from . import defect_register
            defects = defect_register.load(root=root)["defects"]
        except Exception as exc:
            defects = []
            errors.append("The defect register cannot be read (%s: %s): open defects are not counted against the "
                          "stages." % (type(exc).__name__, exc))
    stages = []
    for key, name, required in STAGES:
        named = [r for r in rows if isinstance(r, dict) and evidence_stage(r) == key]
        evidence, holding = [], []
        for row in named:
            holds, why = row_holds(row)
            claim = str((row.get("evidence_for") or {}).get("claim") or "").strip()
            entry = dict(id=row.get("id"), title="; ".join(tr._lines(row.get("title"))), claim=claim,
                         outcome=tr.recorded_outcome(row), holds=holds, why_not=why)
            evidence.append(entry)
            if holds:
                holding.append(entry)
        # an open GOF-BLOCKER: a "closed" typed beside a state other than FAILED/SUPERSEDED keeps the entry open
        # (defect_register.is_open; the dashboard review of Oct 5, 2026, finding L7)
        from . import defect_register
        blocking = [dict(id=d.get("id"), title=d.get("title"), state=d.get("state"))
                    for d in defects if isinstance(d, dict) and d.get("stage") == key
                    and d.get("severity") == "GOF-BLOCKER" and defect_register.is_open(d)]
        demonstrated = bool(holding) and not blocking and not errors
        reasons = []
        if not named:
            reasons.append("No test registry row names this stage as evidence.")
        elif not holding:
            reasons.append("%d %s this stage as evidence and none holds: %s." % (
                len(named), "row names" if len(named) == 1 else "rows name",
                "; ".join("%s (%s)" % (e["id"], e["why_not"]) for e in evidence)))
        if blocking:
            reasons.append("%d open GOF-BLOCKER %s this stage: %s." % (
                len(blocking), "defect names" if len(blocking) == 1 else "defects name",
                ", ".join("%s (%s)" % (b["id"], b["state"]) for b in blocking)))
        stages.append(dict(key=key, name=name, required_evidence=required,
                           state="DEMONSTRATED" if demonstrated else "NOT DEMONSTRATED", reasons=reasons,
                           evidence_rows=evidence, blocking_defects=blocking))
    final = next(s for s in stages if s["key"] == FINAL_STAGE)
    abilities = [e["claim"] or e["title"] for e in final["evidence_rows"] if e["holds"]] \
        if final["state"] == "DEMONSTRATED" else []
    shown = [s["name"] for s in stages if s["state"] == "DEMONSTRATED"]
    out.update(stages=stages, gain_of_function_demonstrated=final["state"] == "DEMONSTRATED",
               demonstrated_abilities=abilities, errors=errors,
               headline=("Gain of function demonstrated: %s." % "; ".join(abilities) if abilities else
                         "No gain of function demonstrated."),
               sentence=("No gain of function demonstrated. %d of 5 stages are demonstrated%s." % (
                   len(shown), (": " + ", ".join(shown)) if shown else "") if not abilities else
                         "Gain of function demonstrated; %d of 5 stages are demonstrated." % len(shown)),
               how_a_stage_is_shown=("A stage is DEMONSTRATED when a test registry row names it in evidence_for, the "
                                     "row's outcome is PASS written by a runner, every structured reading of its "
                                     "result says yes to both judgements, and no open GOF-BLOCKER defect names the "
                                     "stage."))
    return out
