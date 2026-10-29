# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Selected-run disclosures for the explainer; standard library only.

Recorded settings describe the selected run, never current parser defaults or an
unmerged repair. The coordinator status page owns qualification and next actions.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path


# Only flags this checkout's train_mimo parses (each is also in build_explainer.PUBLISHED_ARGS).
# --heads-memory-constant, --heads-score-bound, --gaze-copy-centre and --liveness-centre-pace were
# listed here from 2026-09-27 to 2026-10-02 while they existed only on unmerged branches, so every
# production run showed them as "Not recorded"; the merge that adds one to train_mimo adds it back.
REPAIR_SETTINGS = (
    ("liveness_repair", "Reporting and repair mode for inactive network units"),
    ("liveness_rule", "Rule for automatically replacing inactive network units"),
    ("persistent_memory_repair", "Repair the learned recall of stored experiences"),
    ("persistent_memory_constant_return", "Rate for shrinking memory's accumulated constant; 0 retains it"),
    ("motor_input_fold", "Adjust movement input without changing its initial output"),
    ("motor_asked_change", "Give the motor encoder each handle's asked change (command minus achieved pose)"),
    ("self_percept_release", "Repair recall used to predict the body's own condition"),
)


def read_object(path):
    try:
        with open(path, encoding="utf-8") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def same_path(a, b):
    return bool(a and b) and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def selected_production_run(saved):
    """Selection wins over experimental stats, even while new args are absent.

    The caller then waits/fails for that selection rather than silently replacing
    production with whichever experimental folder last wrote a stats line.
    """
    selection = Path(saved) / "training_continuation.json"
    if not selection.exists():
        return None
    active = read_object(selection).get("active_run")
    if isinstance(active, str) and active:
        return os.path.abspath(active)
    raise ValueError("Production selection is unreadable or missing active_run; retaining previous page")


def setting_record(args, name):
    if name not in args:
        return {"recorded": False, "value": None, "label": "Not recorded in this run"}
    value = args[name]
    if value is None:
        label = "Recorded as null; no setting supplied"
    elif value is False or value == "off":
        label = "Off"
    elif value is True or value == "on":
        label = "On"
    else:
        label = str(value)
    if name == "liveness_rule":
        label = {"none": "No automatic replacement (none)",
                 "live_line": "Replace units below the response threshold (live_line)",
                 "varying": "Also replace units whose outputs barely vary (varying)"}.get(value, label)
    return {"recorded": True, "value": value, "label": label}


def local_time_label(timestamp):
    when = datetime.fromtimestamp(timestamp).astimezone()
    zone = {"Pacific Daylight Time": "PDT", "Pacific Standard Time": "PST"}.get(when.tzname(), when.tzname())
    return when.strftime("%b %d, %Y at %I:%M:%S %p ") + zone


def current_state(run, args, stats, saved, source_paths):
    active = read_object(Path(saved) / "training_continuation.json").get("active_run")
    production = same_path(run, active)
    supervisor = read_object(Path(saved) / "supervisor_control.json")
    control = read_object(Path(run) / "control.json")
    paused = production and supervisor.get("cmd") == "pause"
    # a repair run trains under a pause that names it (the relaunch review of Oct 5, 2026, finding F10): its restart is
    # paused, its training is not
    repair_run = bool(paused and supervisor.get("repair_run") and same_path(run, supervisor.get("repair_run")))
    latest = max(stats.get("training", []), key=lambda r: r.get("training_step", -1), default={})
    stamp = latest.get("timestamp")
    measured_at = datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds") if stamp else None
    measured_label = local_time_label(stamp) if stamp else None
    sources = {}
    for path in source_paths:
        p = Path(path)
        try:
            sources[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            sources[p.name] = None
    # a completed segment records its end in stage_done.json (train_mimo writes it when the run reaches --max-steps); a
    # page built for that segment while the supervisor is paused says so, so the paused segment's step and settings are
    # not read as a running production run's (Oct 15, 2026: "identify training as paused at 494025")
    done = read_object(Path(run) / "stage_done.json")
    ended = ({"steps": done.get("steps"), "label": local_time_label(done["finished_at"])
              if isinstance(done.get("finished_at"), (int, float)) else None}
             if done.get("ok") is True and isinstance(done.get("steps"), int) else None)
    return {
        "selected_run": os.path.abspath(run), "production_run": active,
        "is_selected_production": production, "restart_paused": paused, "repair_run": repair_run,
        "run_ended": ended, "supervisor_paused": supervisor.get("cmd") == "pause",
        "supervisor_command": supervisor.get("cmd") if production else None,
        "run_command": control.get("cmd"),
        "latest_training_update": latest.get("training_step"),
        "latest_training_recorded_at": measured_at, "bodies": args.get("n_envs"),
        "latest_training_recorded_label": measured_label,
        "repair_settings": {name: setting_record(args, name) for name, _ in REPAIR_SETTINGS},
        "source_sha256": sources,
    }


def current_state_html(state):
    esc = lambda value: html.escape(str(value), quote=True)
    ended = state.get("run_ended")
    if state["is_selected_production"]:
        title = (("Repair run: bounded; it ended at model step %s and is not restarted automatically"
                  % "{:,}".format(ended["steps"]) if ended else
                  "Repair run: bounded, with no automatic restart when it ends") if state.get("repair_run") else
                 "Production restart is paused" if state["restart_paused"] else "Selected production run")
    elif state.get("run_ended"):
        title = "Completed run segment: it ended at model step %s" % "{:,}".format(state["run_ended"]["steps"])
    else:
        title = "Selected experiment or earlier run segment"
    step = state["latest_training_update"]
    measurement = f"Latest recorded training update: {step:,}." if step is not None else "No training update is recorded in this run."
    if state["latest_training_recorded_at"]:
        measurement += " Recorded " + esc(state["latest_training_recorded_label"]) + "."
    if state["bodies"] is not None:
        measurement += f" {esc(state['bodies'])} simulated bodies."
    if state.get("run_ended"):
        ended = state["run_ended"]
        measurement += (" This run's stage_done.json records its end at model step %s%s." % (
            "{:,}".format(ended["steps"]), (" on " + esc(ended["label"])) if ended.get("label") else ""))
        if state.get("supervisor_paused") and not state["is_selected_production"]:
            measurement += (" The production supervisor is paused: its control file "
                            "(Saved/mimo_bridge/supervisor_control.json) reads pause, and its production record still "
                            "names %s, another run this page does not show." % esc(
                                os.path.basename(str(state.get("production_run") or "")) or "no run"))
    rows = "".join(
        f"<tr><td>{esc(label)}<br><code>--{esc(name.replace('_', '-'))}</code></td>"
        f"<td>{esc(state['repair_settings'][name]['label'])}</td></tr>"
        for name, label in REPAIR_SETTINGS
    )
    # Decisions and test progress have one maintained home, the development dashboard's Now page, computed from the
    # defect register and the test registry on every request. A typed, dated decision note stood here until Oct 7, 2026
    # ("H84 decision, September 28"), and its link opened a status page last rewritten on Oct 3 (the decided design of
    # the Architecture tab, D-E7); neither is restored here.
    return (
        '<aside class="run-state" aria-label="Selected run and repair status">'
        f'<h2>{title}</h2><p>{measurement}</p>'
        '<p>The diagrams explain the selected run. A refreshed explanation is not '
        'new training or evidence that a repair worked. '
        '<a href="http://127.0.0.1:8766/#now">'
        'Current repair states, open defects and next tests: the development dashboard\'s Now page</a>.</p>'
        '<details><summary>Which repair settings were actually recorded for this run?</summary>'
        '<p>These values come from this run\'s saved settings. “Not recorded” does not '
        'mean “off”: the running code\'s default at that time must be checked. A repair '
        'being tested elsewhere does not change these values.</p>'
        '<table><thead><tr><th>Setting and purpose</th><th>Recorded value</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></details></aside>'
    )
