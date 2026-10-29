# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The dashboard's JSON routes of Oct 5, 2026: the data the Tests screen, the Terms and definitions tab, the
Architecture popup and the top of the dashboard read (the operator's recovery directive, sections 5, 7, 9, 10, 12, 13).

Every route is read-only and computed on the request (kept for KEEP_S seconds per route, so a page that asks several
times a second does not start git and scan the processes each time). Two folders are named on every call:
`data_root` holds the test registry and Saved/ (the runs, the supervisor's files); `code_root` holds the component
manifest, the QA log, the defect register and the code's git history (this module's own checkout when None). They
are one folder in production; a second dashboard started from a worktree reads production's data with its own code.

    GET /api/tests            test_registry.dashboard_view (served by development_dashboard.py; ?view=sections puts
                              each row inside its section)
    GET /api/registry_audit   test_runner.audit: live test processes against running rows, both directions
    GET /api/components       component_manifest.status: every component with its generated fields and the check;
                              ?id=PLANNER_SEARCH limits it to one component; ?history=0 leaves out last code change
    GET /api/defects          defect_register.view: the register with ages, the 12-hour and 24-hour rules, problems
    GET /api/machine_mode     {"now": machine_mode.mode, "week": machine_mode.week}
    GET /api/scoreboard       gof_scoreboard.scoreboard: the five stages
    GET /api/names            names(): the canonical names and their definitions, one list per kind
    GET /api/overview         overview(): the top of the dashboard in one read
    GET /api/investigations   elimination.view: every root-cause investigation's process-of-elimination diagram, its
                              boxes computed from the test registry's rows on this read (the style guide, the operator's
                              rule of Oct 5, 2026); ?id=motor_handle_reach limits it to one investigation
"""
from __future__ import annotations

from pathlib import Path
import threading
import time

from . import test_registry as tr

ROUTES = ("/api/registry_audit", "/api/components", "/api/defects", "/api/machine_mode", "/api/scoreboard",
          "/api/names", "/api/overview", "/api/investigations")
KEEP_S = 8.0
_KEPT = {}
_LOCK = threading.Lock()


def _kept(key, build):
    now = time.time()
    with _LOCK:
        found = _KEPT.get(key)
        if found and now - found[0] < KEEP_S:
            return found[1]
    value = build()
    with _LOCK:
        if len(_KEPT) > 64:
            _KEPT.clear()
        _KEPT[key] = (time.time(), value)
    return value


def names(code_root=None) -> dict:
    """GET /api/names: every canonical name with its definition, for the Terms and definitions tab and "The names
    used on this page". components come from the component manifest (id, kind, name, does, aliases: the aliases are
    for search only and are refused in operator-facing text); the fixed vocabularies come from the modules that
    compute with them."""
    from . import component_manifest, defect_register, gof_scoreboard, machine_mode
    # directive section 7B: definitions load expanded; one button toggles between the two labels
    out = dict(schema=1, read_at_text=tr.clock(), expanded_by_default=True,
               toggle_labels=dict(when_expanded="Collapse all", when_collapsed="Expand all"))
    try:
        data = component_manifest.load(code_root)
        out["components"] = [dict(id=c.get("id"), kind=c.get("kind"), name=c.get("name"), definition=c.get("does"),
                                  aliases=c.get("aliases") or [], blocks=c.get("blocks") or [],
                                  flags=c.get("flags") or [])
                             for c in data["components"] if isinstance(c, dict)]
        out["forbidden_aliases"] = [dict(alias=alias, canonical=ids)
                                    for alias, ids in component_manifest.aliases(code_root)]
    except component_manifest.ManifestError as exc:
        out.update(components=[], forbidden_aliases=[], components_error=str(exc))
    out.update(
        repair_states=[dict(name=s, definition=defect_register.STATE_MEANINGS[s]) for s in defect_register.STATES],
        fixed_rule="Only VERIFIED may be called fixed.",
        status_form="A status is written as NAME: STATE, with NAME a component or defect id and STATE one of the "
                    "eight repair states.",
        severities=list(defect_register.SEVERITIES),
        severity_meanings=defect_register.severity_meanings(code_root),
        test_sections=[dict(name=s, definition=tr.SECTION_MEANINGS[s]) for s in tr.SECTIONS],
        test_outcomes=[dict(name=s, definition=tr.OUTCOME_MEANINGS[s]) for s in tr.OUTCOMES],
        resource_classes=[dict(name=k, definition=v) for k, v in tr.RESOURCE_CLASSES.items()],
        process_states=[dict(name=k, definition=v) for k, v in tr.PROCESS_STATES.items()],
        machine_modes=[dict(name=m, definition=machine_mode.MODE_MEANINGS[m]) for m in machine_mode.MODES],
        scoreboard_stages=[dict(name=key, plain_name=name, definition=required)
                           for key, name, required in gof_scoreboard.STAGES],
        reading_judgements=dict(statistically_clear=list(tr.STATISTICAL_ANSWERS), gof_material=list(tr.GOF_ANSWERS)))
    return out


def overview(data_root=None, code_root=None) -> dict:
    """GET /api/overview: the top of the dashboard. machine_mode (exactly one mode, with its sentence); scoreboard
    (the headline and each stage's state); top_blockers (the open GOF-BLOCKER defects in the register's order: the
    ones idle 12 hours or more first, at most 5); status_lines ("ID: STATE" for every open defect, generated from
    the register); errors [{source, text}]: every finding the pages must show as an error (the audit's mismatches,
    the manifest check's findings, the register's problems, the mode's errors)."""
    from . import component_manifest, defect_register, gof_scoreboard, machine_mode, test_runner
    data_root = Path(data_root or tr.REPO)
    errors = []

    def guarded(source, build, fallback):
        try:
            return build()
        except Exception as exc:
            errors.append(dict(source=source, text="%s could not be computed (%s: %s)." % (source, type(exc).__name__,
                                                                                          exc)))
            return fallback
    audit = guarded("The audit of test processes", lambda: test_runner.audit(root=data_root, code_root=code_root),
                    dict(available=False, mismatches=[], running_rows=[], test_processes=[], development_runs=[]))
    for found in audit.get("mismatches") or []:
        errors.append(dict(source="test registry audit", text=found.get("text")))
    mode = guarded("The machine mode", lambda: machine_mode.mode(data_root, audit_view=audit),
                   dict(mode=None, sentence="The machine mode could not be computed.", errors=[]))
    for text in mode.get("errors") or []:
        errors.append(dict(source="machine mode", text=text))
    board = guarded("The scoreboard", lambda: gof_scoreboard.scoreboard(data_root, code_root),
                    dict(headline="The scoreboard could not be computed.", stages=[], errors=[]))
    for text in board.get("errors") or []:
        errors.append(dict(source="scoreboard", text=text))
    defects = guarded("The defect register", lambda: defect_register.view(code_root, data_root),
                      dict(defects=[], problems=[], top=[], holding=[], status_lines=[], sentence=""))
    for text in defects.get("problems") or []:
        errors.append(dict(source="defect register", text=text))
    check = guarded("The component manifest's check", lambda: component_manifest.check(code_root),
                    dict(ok=False, problems=[], sentence="The component manifest's check could not run."))
    for text in check.get("problems") or []:
        errors.append(dict(source="component manifest", text=text))
    blockers = [d for d in defects.get("defects") or [] if d.get("open") and d.get("severity") == "GOF-BLOCKER"]
    return dict(
        schema=1, read_at_text=tr.clock(),
        machine_mode=dict(mode=mode.get("mode"), sentence=mode.get("sentence"), reason=mode.get("reason"),
                          since_text=mode.get("since_text"), held_at_step=mode.get("held_at_step"),
                          active_run=mode.get("active_run"), live_tests=mode.get("live_tests"),
                          development_runs=mode.get("development_runs")),
        scoreboard=dict(headline=board.get("headline"), sentence=board.get("sentence"), question=board.get("question"),
                        how_a_stage_is_shown=board.get("how_a_stage_is_shown"),
                        gain_of_function_demonstrated=board.get("gain_of_function_demonstrated", False),
                        stages=[dict(key=s["key"], name=s["name"], state=s["state"], reasons=s["reasons"],
                                     required_evidence=s.get("required_evidence"),
                                     blocking_defects=[dict(id=b.get("id"), state=b.get("state"))
                                                       for b in s.get("blocking_defects") or []])
                                for s in board.get("stages") or []]),
        defects=dict(sentence=defects.get("sentence"), top=defects.get("top"), holding=defects.get("holding"),
                     rule=defects.get("rule"), idle_meaning=defects.get("idle_meaning")),
        top_blockers=[dict(id=d.get("id"), title=d.get("title"), component=d.get("component"), state=d.get("state"),
                           age_text=d.get("age_text"), idle_text=d.get("idle_text"), idle_short=d.get("idle_short"),
                           top=d.get("top"),
                           needs_explanation=d.get("needs_explanation"), next_action=d.get("next_action"),
                           waits_on=d.get("waits_on"), owner=d.get("owner")) for d in blockers[:5]],
        open_blockers=len(blockers),
        status_lines=defects.get("status_lines") or [],
        tests=dict(audit_sentence=audit.get("sentence"), running=len(audit.get("running_rows") or []),
                   test_processes=len(audit.get("test_processes") or []),
                   development_runs=len(audit.get("development_runs") or []),
                   unverified=len(audit.get("unverified") or [])),
        manifest=dict(ok=check.get("ok"), sentence=check.get("sentence")),
        errors=errors, error_count=len(errors))


def respond(path, query=None, data_root=None, code_root=None) -> dict:
    """The JSON of one of ROUTES. `query` is the request's parsed query ({name: [values]}). Raises KeyError for a
    path that is not a route."""
    from . import component_manifest, defect_register, gof_scoreboard, machine_mode, test_runner
    query = query or {}
    data_root = Path(data_root or tr.REPO)
    key = (path, str(data_root), str(code_root), tuple(sorted((k, tuple(v)) for k, v in query.items())))
    if path == "/api/registry_audit":
        return _kept(key, lambda: test_runner.audit(root=data_root, code_root=code_root))
    if path == "/api/components":
        only = (query.get("id") or [None])[0]
        history = (query.get("history") or ["1"])[0] != "0"
        return _kept(key, lambda: component_manifest.status(code_root, data_root, only=only, with_history=history))
    if path == "/api/defects":
        return _kept(key, lambda: defect_register.view(code_root, data_root))
    if path == "/api/machine_mode":
        return _kept(key, lambda: dict(now=machine_mode.mode(data_root), week=machine_mode.week(data_root)))
    if path == "/api/scoreboard":
        return _kept(key, lambda: gof_scoreboard.scoreboard(data_root, code_root))
    if path == "/api/names":
        return _kept(key, lambda: names(code_root))
    if path == "/api/overview":
        return _kept(key, lambda: overview(data_root, code_root))
    if path == "/api/investigations":
        from . import elimination
        only = (query.get("id") or [None])[0]
        return _kept(key, lambda: elimination.view(data_root=data_root, code_root=code_root, only=only))
    raise KeyError(path)
