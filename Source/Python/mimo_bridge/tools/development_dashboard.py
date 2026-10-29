# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Local development dashboard: live training state and versioned evidence.

Serve on loopback, read measurements without loading models, and retain the old
HTML pages. Evidence changes require a matching receipt and explicit review;
an offline result cannot mark an ability acquired by the running model.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit

REPO = Path(__file__).resolve().parents[4]
ASSETS = Path(__file__).parent
SAVED = REPO / 'Saved/mimo_bridge'
REGISTRY = REPO / 'docs/development_dashboard/evidence.json'


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON: {value}')
    def finite(value):
        number=float(value)
        if not math.isfinite(number):reject(value)
        return number
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject,parse_float=finite)


def confined(root, path):
    from ..repo_paths import localize   # a path recorded inside D:\UEMA_Research names this checkout's copy
    resolved = (root / localize(path)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Path outside the project')
    return resolved


def file_url(root, path):
    return '/files/' + quote(confined(root, path).relative_to(root).as_posix(), safe='/')


def checkpoint_reference(root,path):
    """Read ancestry on explicitly registered storage; never expand HTTP access."""
    try:return confined(root,path)
    except ValueError:
        target=Path(path).resolve()
        catalog=root/'Saved/mimo_bridge/checkpoint_storage_roots.json'
        roots=read_json(catalog)['roots'] if catalog.is_file() else []
        if not any(target.is_relative_to(Path(base).resolve()) for base in roots):
            raise ValueError('Unregistered checkpoint storage')
        if not re.fullmatch(r'ckpt_step_\d+(?:_.*)?',target.name) or target.parent.name!='checkpoints':
            raise ValueError('Not a checkpoint ancestry reference')
        return target


# THE TAIL SIZES THE NEWEST TRAINING RECORD IS LOOKED FOR IN, in bytes, tried in turn until one holds a
# complete 'mimo' record. One size of 256,000 was enough while a record stayed under it: H84's last 59 records
# (4 run folders up to Oct 3, 2026) were 182,761 to 234,927 bytes. The swap test's added readings and
# controllability's shadow gate add 36,215 bytes to each (33,344 and 2,871), which puts 7 of those 59 past
# 256,000 bytes: the tail then began inside the record's own line, no complete record was found, and the
# page would have read no newest record for that report window.
TRAINING_RECORD_TAILS = (256_000, 1_024_000, 4_096_000)


def last_training_record(path):
    """Read a bounded tail; incomplete writes do not replace the last record. The tail grows through
    TRAINING_RECORD_TAILS until it holds one complete 'mimo' record or covers the whole file."""
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            for tail in TRAINING_RECORD_TAILS:
                stream.seek(max(0, size - tail))
                lines = stream.read().decode('utf-8', errors='replace').splitlines()
                for line in reversed(lines):
                    try:
                        row = json.loads(line)
                        if row.get('type') == 'mimo' and all(
                                isinstance(row.get(k), (int, float)) and math.isfinite(row[k])
                                for k in ('step', 'timestamp')):
                            return row
                    except (ValueError, AttributeError, TypeError):
                        continue
                if tail >= size:
                    break
    except OSError:
        pass
    return {}


# The launch settings the Now tab lists: H84's behaviour-deciding flags, then every flag the
# gain-of-function build adds (DESIGN.md section 2; each defaults to the old behaviour, so a
# run launched before the build reads "not present at launch").
SETTINGS = [
    ('intention', 'the planner picks a remembered best-felt model step as its goal and plans a command toward it'),
    ('intention_every', 'model steps between planning events'),
    ('imagination', 'a training lesson on predictions chained several steps ahead; it never moves the body'),
    ('plan_horizon', 'how far ahead the planner may look ("earned": only as far as its predictions beat doing nothing)'),
    ('babble_noise', 'size of random exploration added to commands not written by the planner'),
    ('smooth_actions', 'share of the previous command blended into the next one (0 = none)'),
    ('motor_hold', 'model steps each command stays in force'),
    ('decoder_kp', 'stiffness with which a commanded posture is pursued'),
    ('wellness_mode', 'which wellness formula scores each model step'),
    ('wellness_coherence', 'which reading of the body map the coherence term of wellness carries: extent = the mean of the map\'s co-movement numbers (how often a handle\'s command change and proprioception\'s change fall in the same model step; not causation), certainty = how far those numbers sit from the middle of their range'),
    ('consequence_w', 'weight of the consequence term in wellness'),
    ('self_touch_wellness', 'weight of the self-touch term in wellness (0 = it pays nothing)'),
    ('anticipation_gaze', 'a hand-coded "look at the consequence" lesson (DESIGN.md D14: off from the gain-of-function deploy of 2026-09-23)'),
    ('entity_self', 'the self-model that watches named things (dark = observes and changes no behaviour)'),
    ('self_value_weight', 'weight the self-model reader may add to a goal once a link is proven'),
    ('hierarchical_self', 'the older slot-table self-model'),
    ('top_down_gate', 'blends up to half of the model\'s own prediction into what behaviour consumes'),
    ('residual_prediction', 'the prediction heads predict the change and add it to the present'),
    ('rest_control', 'who decides when the body rests'),
    ('toy_hand_off', 'the adult can hand the toy over so it stays in the baby\'s hand'),
    ('d_model', 'width of the attention core (H84: 192; the earlier lineages: 128)'),
    ('core_blocks', 'number of attention blocks in the core'),
    ('n_envs', 'bodies trained at once, each in its own world'),
    ('causal_chain', 'logs goal, predicted outcome, predicted wellness, the alternatives and the realised result of each plan'),
    ('efference_heads', 'two heads given the executed command directly predict where each handle and the eye go'),
    ('efference_weight', 'loss weight of each efference head'),
    ('vestibular_sense', 'vestibular, the body\'s load and balance readings, takes the intermind placeholder\'s slot'),
    ('wellness_rank', 'goals ranked on channels put on one scale by their own spread (normalized) or on the raw sum'),
    ('wellness_weights', 'priority weight of each wellness channel in the ranking'),
    ('controllability', 'wellness channel: the body did what the model expected its command to do'),
    ('learning_progress', 'wellness channel: prediction errors falling, measured against "nothing changes"'),
    ('curiosity_mode', 'curiosity multiplies the goal value (multiply) or adds a channel (additive)'),
    ('consequence_self_contact', 'whether touching its own body counts as a consequence (exclude = world contact only)'),
    ('contact_head', 'a head predicts which skin patches are about to be touched, given the command'),
    ('contact_control_wellness', 'weight of wellness paid for touches predicted before they happened (0 = off)'),
    ('motor_synergy', 'learns a small map from a compact intention to many joints'),
    ('synergy_planner', 'lets the planner use the learned movement map; off lets the map learn first'),
    ('synergy_dim', 'size of that compact intention'),
    ('synergy_babble', 'exploration noise added in the compact intention instead of per joint'),
    ('synergy_residual_l2', 'how strongly the map\'s per-joint remainder is kept small'),
    ('smooth_source', 'the smoother blends with the previous executed command (executed) or with proposals (proposal)'),
    ('next_frame_norm', 'each sense\'s prediction error divided by its own "nothing changes" error (persistence)'),
    ('liveness_repair', 'reports responding units, or also revives units with no response in the observation window'),
    ('liveness_rule', 'which units the repair revives: live_line = input never above -3; varying = also units whose output did not vary; none = no unit (units revived before keep their routing and learning)'),
    ('resume_env_settings', 'keeps saved world settings or explicitly migrates supported settings from this launch'),
]


def settings_for(args):
    out = []
    for key, meaning in SETTINGS:
        value = args.get(key) if key in args else None
        text = ('not present at launch (the default keeps the old behaviour)' if key not in args else
                'not set' if value is None else 'on' if value is True else 'off' if value is False else str(value))
        out.append(dict(flag='--'+key.replace('_','-'), value=text, meaning=meaning))
    return out


def _record_fields(row, args, now):
    """The fields of the training state read from a run's newest record and launch settings."""
    return dict(step=row.get('step'), record_timestamp=row.get('timestamp'),
                steps_per_second=row.get('model_steps_per_s'),
                caregiver=row.get('caregiver',[]),body_orientation=row.get('body_orientation',[]),
                motor_contract=row.get('motor_contract'),nan_steps=row.get('nan_steps'),
                settings=settings_for(args),
                record_age_seconds=max(0, now-row['timestamp']) if row else None,
                # kept out of the JSON (build_state pops them): the full newest record and
                # the launch settings, read once per request
                _record=row, _args=args)


def production_training_state(root, now):
    saved = root / 'Saved/mimo_bridge'
    try:
        state = read_json(saved/'training_continuation.json')
        run = confined(root, state['active_run'])
        row = last_training_record(run/'replay_training_stats.jsonl')
        args=read_json(run/'args.json') if (run/'args.json').exists() else {}
        timestamp = state.get('timestamp')
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError('Missing supervisor timestamp')
        return dict(run=run.relative_to(root).as_posix(), run_name=run.name,
                    supervisor_status=state.get('status', 'unknown'),
                    supervisor_timestamp=timestamp, pause_reason=state.get('pause_reason'),
                    trainer_pids=state.get('trainer_pids'),
                    source=file_url(root, saved/'training_continuation.json'),
                    **_record_fields(row, args, now))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(run=None, run_name=None, supervisor_status='unknown',
                    record_timestamp=None, supervisor_timestamp=None, step=None,
                    pause_reason=None, record_age_seconds=None)


def production_owns_the_page(training, now):
    """Production keeps the page while its supervisor reports "training" in a record under 3 minutes old, whatever
    the age of the run's newest training record (a relaunch before its first record reads "not confirmed" as it did
    before the experiment view)."""
    sup = training.get('supervisor_timestamp')
    return (str(training.get('supervisor_status')) == 'training' and isinstance(sup, (int, float))
            and now-sup < 180)


def experiment_training_state(root, now, production):
    """THE EXPERIMENT VIEW (Oct 13, 2026): the live experiment run named by Saved/mimo_bridge/experiment_live.json,
    read from the copies tools/experiment_mirror.py keeps in Saved/mimo_bridge/experiment_mirror/<run name>/ (inside
    the project, so confined() and the /files/ rule stay as they are). None unless the pointer's owner process is
    alive as recorded (experiment_mirror.live_view). In this view supervisor_timestamp is the mirror's last pass (when
    it last read the owner state and copied), so the page's 3-minute line judges the copies' freshness."""
    from .experiment_mirror import live_view
    view = live_view(root)
    if not view:
        return None
    run = confined(root, view['mirror'])
    row = last_training_record(run/'replay_training_stats.jsonl')
    args = read_json(run/'args.json') if (run/'args.json').exists() else {}
    status = ('training' if view['phase'] == 'training' and view['trainer_alive'] else
              f"experiment owner phase {view['phase']}" + ('' if view['trainer_alive'] else ', trainer process not alive'))
    return dict(run=run.relative_to(root).as_posix(), run_name=run.name, view='experiment',
                supervisor_status=status, supervisor_timestamp=view['mirror_pass_at'], pause_reason=None,
                trainer_pids=[view['trainer_pid']] if view['trainer_alive'] else [],
                source=file_url(root, run/'mirror_status.json') if (run/'mirror_status.json').is_file() else None,
                experiment=dict(source_run=view['source_run'], owner=view['owner'], owner_pid=view['owner_pid'],
                                owner_alive=view['owner_alive'], trainer_pid=view['trainer_pid'],
                                trainer_alive=view['trainer_alive'], phase=view['phase'],
                                started_at=view['started_at'], mirror=run.relative_to(root).as_posix(),
                                mirror_pass_at=view['mirror_pass_at'], mirror_pid=view['mirror_pid'],
                                mirror_alive=view['mirror_alive'], project_parent=view['project_parent'],
                                production_run=production.get('run_name'),
                                production_run_path=production.get('run'),
                                production_record_timestamp=production.get('record_timestamp'),
                                production_status=production.get('supervisor_status'),
                                production_pause_reason=production.get('pause_reason')),
                **_record_fields(row, args, now))


def training_state(root, now):
    """The run the page describes: production's active run while its supervisor reports training
    (production_owns_the_page); otherwise the live experiment run when one is recorded (experiment_training_state);
    otherwise production's paused or ended run, as before."""
    production = production_training_state(root, now)
    if production_owns_the_page(production, now):
        return production
    try:
        experiment = experiment_training_state(root, now, production)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Dashboard: the experiment run could not be read: {type(exc).__name__}: {exc}', flush=True)
        experiment = None
    return experiment or production


_MONTHS = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')


def _zone_name(moment):
    """The zone as the operator reads it: 'PDT'. Windows spells it out ('Pacific Daylight Time')."""
    name = time.strftime('%Z', moment) or ''
    words = name.split()
    return ''.join(w[0] for w in words).upper() if len(words) > 1 else name


def clock(stamp, date=True):
    """A clock time as the page prints it: 'Sep 24, 2026, 6:05 PM PDT' - AM/PM in this computer's
    time zone, never a 24-hour clock (operator 2026-09-23). date=False gives '6:05 PM PDT'."""
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        return 'an unrecorded time'
    moment = time.localtime(stamp)
    text = f"{moment.tm_hour % 12 or 12}:{moment.tm_min:02d} {'AM' if moment.tm_hour < 12 else 'PM'} {_zone_name(moment)}".strip()
    return f"{_MONTHS[moment.tm_mon - 1]} {moment.tm_mday}, {moment.tm_year}, {text}" if date else text


def local_stamp(text):
    """A 24-hour local time written by a tool ('2026-09-23 00:40', seconds optional) as a timestamp,
    so the page can print it in AM/PM; None when the text is not such a time."""
    for layout in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
        try:
            return datetime.strptime(str(text).strip(), layout).timestamp()
        except (TypeError, ValueError):
            continue
    return None


def _when(stamp):
    return clock(stamp)


_CLOCK_24 = re.compile(r'\b(20\d\d-\d\d-\d\d)[ T](\d\d:\d\d(?::\d\d)?)\b')


def am_pm_text(text):
    """A 24-hour local time inside a sentence written elsewhere ('2026-09-15 05:29', as the
    experiment board and the 2026-09-13 work registry write them) in AM/PM with the zone."""
    def swap(match):
        stamp = local_stamp(f"{match.group(1)} {match.group(2)}")
        return clock(stamp) if stamp is not None else match.group(0)
    return _CLOCK_24.sub(swap, text)


def am_pm_tree(obj):
    """am_pm_text over every sentence of a page record, leaving identifiers and paths as they are."""
    from .plain_numbers import _LITERAL_KEYS
    if isinstance(obj, dict):
        return {k: (v if k in _LITERAL_KEYS else am_pm_tree(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [am_pm_tree(v) for v in obj]
    return am_pm_text(obj) if isinstance(obj, str) else obj


def _age_words(seconds):
    seconds=max(0,int(seconds))
    return (f'{seconds} seconds' if seconds<120 else f'{seconds//60} minutes' if seconds<7200
            else f'{seconds//3600} hours')


def training_verdict(root, training, now):
    """The training state in plain words, verdict first: running, paused on purpose, stopped
    by a stop rule (with its numbers), or unconfirmed. Reads supervisor_control.json, the
    supervisor's record, and the active run's stop_reason.json and stage_done.json."""
    from .lineage_series import motor_watch_note, plain_stop_reason, stop_writer
    saved=root/'Saved/mimo_bridge'
    control=read_json_or(saved/'supervisor_control.json',{})
    run=(root/training['run']) if training.get('run') else None
    stop=read_json_or(run/'stop_reason.json',None) if run else None
    done=read_json_or(run/'stage_done.json',None) if run else None
    name=training.get('run_name') or 'the active run'
    sup=training.get('supervisor_timestamp');rec_age=training.get('record_age_seconds')
    sup_age=(now-sup) if isinstance(sup,(int,float)) else None
    status=str(training.get('supervisor_status') or 'unknown')
    running=(status=='training' and sup_age is not None and sup_age<180 and rec_age is not None and rec_age<600)
    if training.get('view')=='experiment':
        return experiment_verdict(root,training,control,now,running,sup_age,rec_age)
    if isinstance(stop,dict) and isinstance(training.get('record_timestamp'),(int,float)) and \
            isinstance(stop.get('timestamp'),(int,float)) and training['record_timestamp']>stop['timestamp']+600:
        stop=None      # an older stop the run has since written past
    facts=[]
    if isinstance(stop,dict):
        writer=stop_writer(stop)
        facts.append(f"{writer[0].upper()+writer[1:]} "
                     f"stopped {name} at {_when(stop.get('timestamp'))}: {plain_stop_reason(stop.get('reason'))}.")
    if isinstance(done,dict) and done.get('steps'):
        # finished_at_human is the trainer's 24-hour text; the timestamp prints in AM/PM
        written=(_when(done['finished_at']) if isinstance(done.get('finished_at'),(int,float)) else
                 _when(local_stamp(done.get('finished_at_human'))) if local_stamp(done.get('finished_at_human')) else
                 done.get('finished_at_human') or _when(None))
        facts.append(f"Its final checkpoint is at lineage step {int(done['steps']):,}, written {written}.")
    if running:
        sps=training.get('steps_per_second')
        state='running'
        sentence=(f"Training is running: {name} is at lineage step {int(training['step']):,}; its newest record is "
                  f"{_age_words(rec_age)} old" + (f" and it runs {sps:.2f} model steps per second" if isinstance(sps,(int,float)) else '') + '.')
        facts=[]
    elif control.get('cmd')=='pause':
        state='paused'
        reason=str(control.get('reason') or '').strip()
        sentence='Training is paused on purpose'+(f': {reason}' if reason else ' (supervisor_control.json asks the supervisor to pause; no reason was written)')+'.'
    elif isinstance(stop,dict):
        state='stopped'
        sentence=(f"Training is stopped: {plain_stop_reason(stop.get('reason'))}. The supervisor waits for a repair "
                  f"before it resumes.")
        facts=facts[1:]
    elif training.get('pause_reason'):
        state='paused'
        sentence=f"Training is paused: the supervisor reports \"{training['pause_reason']}\"."
    elif not training.get('run'):
        state='unknown'
        sentence='Training state is unknown: the supervisor record is missing or unreadable.'
    else:
        state='unconfirmed'
        sentence=(f"Training activity is not confirmed: the supervisor reports \"{status}\""
                  + (f", its record is {_age_words(sup_age)} old" if sup_age is not None else '')
                  + (f", and the newest training record is {_age_words(rec_age)} old" if rec_age is not None else '') + '.')
    # the motor fixes kill-line watcher's waiting or unconfirmed stop, in every state (a stop that waits
    # on a hold while training runs is the case that needs a reader most)
    watch=motor_watch_note(saved,now)
    if watch:
        facts.append(watch)
    return dict(state=state,sentence=sentence,facts=facts,
                final_checkpoint_step=done.get('steps') if isinstance(done,dict) else None,
                source=file_url(root,saved/'supervisor_control.json') if (saved/'supervisor_control.json').is_file() else None)


def experiment_verdict(root, training, control, now, running, sup_age, rec_age):
    """The verdict for the experiment view (experiment_training_state): the experiment run named and placed, its owner
    and trainer processes, the mirror's last copy, and production's own state as a fact."""
    from .lineage_series import motor_watch_note, plain_stop_reason, stop_writer
    saved=root/'Saved/mimo_bridge'
    x=training.get('experiment') or {}
    name=training.get('run_name') or 'the experiment run'
    step=training.get('step');stamp=training.get('record_timestamp');sps=training.get('steps_per_second')
    facts=[f"Its owner {x.get('owner') or 'program'} runs as process {x.get('owner_pid')} (alive at this read); its trainer "
           f"process {x.get('trainer_pid')} is " + ('alive.' if x.get('trainer_alive') else 'not alive.'),
           f"The page reads copies that the experiment mirror (tools/experiment_mirror.py, process {x.get('mirror_pid')}) "
           f"writes into {x.get('mirror')}; "
           + (f"its last pass over the run's files was {_age_words(sup_age)} ago" if sup_age is not None
              else 'it has not passed over the run yet') + '.']
    if control.get('cmd')=='pause':
        reason=str(control.get('reason') or '').strip()
        facts.append('Production training is paused on purpose'+(f': {reason}' if reason else '')+'.')
    elif x.get('production_status'):
        facts.append(f"Production's supervisor reports \"{x['production_status']}\" for {x.get('production_run') or 'its run'}.")
    # production's own stop, as training_verdict reads it (an older stop the run has since written past is left out)
    stop=read_json_or(root/x['production_run_path']/'stop_reason.json',None) if x.get('production_run_path') else None
    last=x.get('production_record_timestamp')
    if isinstance(stop,dict) and not (isinstance(last,(int,float)) and isinstance(stop.get('timestamp'),(int,float))
                                      and last>stop['timestamp']+600):
        writer=stop_writer(stop)
        facts.append(f"{writer[0].upper()+writer[1:]} stopped production's {x.get('production_run') or 'run'} at "
                     f"{_when(stop.get('timestamp'))}: {plain_stop_reason(stop.get('reason'))}.")
    parent=x.get('project_parent') or {}
    if isinstance(parent.get('step'),int) and parent.get('run'):
        facts.append(f"Its nearest ancestor checkpoint inside the project is model step {parent['step']:,} of "
                     f"{Path(parent['run']).name}; the abilities and earlier clips on this page follow that ancestry.")
    if running:
        state='running'
        sentence=(f"The experiment run {name} is training outside the production supervisor: it is at model step "
                  f"{int(step):,}; its newest record was written {clock(stamp)}, {_age_words(rec_age)} ago"
                  + (f", and it runs {sps:.2f} model steps per second" if isinstance(sps,(int,float)) else '') + '.')
    else:
        state='unconfirmed'
        sentence=(f"The experiment run {name} is not confirmed as training: its owner records phase \"{x.get('phase')}\""
                  + ('' if x.get('trainer_alive') else ', its trainer process is not alive')
                  + (f", the mirror's last pass is {_age_words(sup_age)} old" if sup_age is not None else ', the mirror has not passed over the run yet')
                  + (f", and its newest record is {_age_words(rec_age)} old" if rec_age is not None else ', and no training record has been copied yet')
                  + '.')
    watch=motor_watch_note(saved,now)
    if watch:
        facts.append(watch)
    return dict(state=state,sentence=sentence,facts=facts,final_checkpoint_step=None,
                source=file_url(root,saved/'supervisor_control.json') if (saved/'supervisor_control.json').is_file() else None)


def ended_experiment_note(root):
    """The training card's fact for an experiment run that is not live (the pointer's run, read from its mirror's
    record): which run, why it is not training, and where its recordings are. None when no experiment was mirrored or
    the experiment is live (the experiment view shows it then)."""
    from .experiment_mirror import live_view, read_pointer, mirror_dir, STATUS
    pointer = read_pointer(root)
    if pointer is None or live_view(root):
        return None
    status = read_json_or(mirror_dir(root, pointer)/STATUS, None)
    if not isinstance(status, dict):
        return None
    name = Path(str(pointer['run'])).name
    head = (f"The experiment run {name} has ended: {status.get('end_reason')}" if status.get('ended') else
            f"The experiment run {name} is not shown as live: its owner or the experiment mirror is not running "
            f"as recorded, so the page cannot confirm its state")
    step = last_training_record(mirror_dir(root, pointer)/'replay_training_stats.jsonl').get('step')
    try:
        clips = len(experiment_clips(root, mirror_dir(root, pointer), None, None))
    except (OSError, ValueError):
        clips = 0
    return (head
            + (f"; its last copied record is model step {int(step):,}" if isinstance(step, (int, float)) else '')
            + (f"; the mirror's last pass was {clock(status['pass_at'])}" if isinstance(status.get('pass_at'), (int, float)) else '')
            + (f". It left {clips} complete recording{'s' if clips != 1 else ''}; Video evidence lists the newest of them "
               f"under its run name, with every run's recordings newest first." if clips else
               ". It left no complete recording in its mirror folder."))


def experiment_changed(root, training):
    """The newest segment's line for the experiment view: the experiment run's start and code, and what it resumed
    (outside the project, so no launch settings are compared here)."""
    from .lineage_series import code_text
    run=root/training['run']
    commit,started,dirty=_launch_commit(run/'replay_training_stats.jsonl')
    resume=str((read_json_or(run/'args.json',{}) or {}).get('resume') or '')
    head=f"Experiment run {training.get('run_name')} started {_when(started)} on {code_text(commit,dirty)}"
    if not resume:
        return head+'.'
    parent=(training.get('experiment') or {}).get('project_parent') or {}
    tail=(f" Its nearest ancestor inside the project is model step {parent['step']:,} of {Path(parent['run']).name}; the "
          f"runs between, outside the project, are not read here, and the Over time tab joins the two."
          if isinstance(parent.get('step'),int) and parent.get('run') else '')
    return head+f"; it resumed the checkpoint {resume}, outside the project."+tail


def _launch_commit(stats):
    """(code commit, launch time, dirty-tree flag or None) of a run's first launch record, from the
    head of its stats file. The trainer does not stamp a dirty tree yet, so the flag is None."""
    try:
        with open(stats,'rb') as stream:
            for line in stream.read(262_144).split(b'\n'):
                if b'"type": "launch"' in line[:48]:
                    row=json.loads(line)
                    return row.get('code_commit'),row.get('timestamp'),row.get('code_dirty')
    except (OSError,ValueError):
        pass
    return None,None,None


def what_changed(root, bounds):
    """The newest segment against the one it resumed from: when, which code, which settings - the
    values IN FORCE (a flag absent from a run ran at its default), each env setting labelled with
    what the resumed worlds did with it (reviews/dashboards.md M8)."""
    from .lineage_series import args_difference, code_text, registry_commits, short_label
    runs=[Path(r) for r in bounds]
    if not runs:
        return None
    active=runs[0];parent=runs[1] if len(runs)>1 else None
    commit,started,dirty=_launch_commit(active/'replay_training_stats.jsonl')
    commit=commit or (registry_commits(root).get(active.name) or [None])[0]
    root_name=runs[-1].name
    label=short_label(active.name,root_name)
    head=f"Segment {label} started {_when(started)} on {code_text(commit,dirty)}"
    if parent is None:
        return head+', from fresh weights.'
    diff=args_difference(read_json_or(active/'args.json',{}),read_json_or(parent/'args.json',{}),commit=commit)
    other=short_label(parent.name,root_name)
    if not diff:
        return head+f'; its launch settings in force match segment {other}.'
    shown='; '.join(d['text'] for d in diff[:6])
    more=f'; and {len(diff)-6} more' if len(diff)>6 else ''
    return head+f'; its launch settings in force differ from segment {other} in: {shown}{more}.'


# plain names for the parts the lead's liveness sweep named (DESIGN.md section 10), in the code's own
# words (tenure_heads["pose"], motor_controller.confidence_head, wellness.approximate,
# persistent_memory.outcome); lineage_series.KNOWN_DEAD_WORDS titles the history charts with the same
# names (test_liveness_part_names.py holds them equal)
PART_NAMES={'tenure_heads.pose.0':'the tenure pose head',
    'contact_head.ctx1':"the contact head's body-and-command input layer",
    'motor_controller.confidence_head.0':"the motor controller's confidence head",
    'wellness.approximate.net.0':"the wellness predictor, first layer",
    'wellness.approximate.net.2':"the wellness predictor, second layer",
    'motor_controller.residual_blocks.0.0':'motor controller, block 1',
    'persistent_memory.outcome.0':"the persistent memory's outcome head",
    'encoder_registry.encoders.motor.pool_attn':'the motor encoder\'s pool (which body parts the motor token reads)',
    'encoder_registry.encoders.tactile.pool_attn':'the tactile encoder\'s pool (which contacts the tactile token reads)',
    # the other layers the production record's liveness block names (review of 2026-09-25, finding 8),
    # read from the code: lem/model/encoders/swin_patch.py AngularPositionEncoding, efference.py
    # EfferenceHeads (body_mlp: Linear, GELU, Linear, GELU, Linear; gaze_in; vision_in),
    # self_state.py GroundedDynamics.local, lem_model.py tenure_heads["order_proj"], the voice head
    'encoder_registry.encoders.vision.angular_pos_enc.mlp.0':"vision encoder's angular position encoding",
    'efference.body_mlp.0':'body head','efference.body_mlp.2':'body head, second layer',
    'efference.gaze_in':'gaze head','efference.vision_in':'gaze copy',
    'self_state.grounded.local.0':"the self-state's grounded dynamics, local layer",
    'tenure_heads.order_proj.0':'the tenure command projection',
    'voice_head.mlp.0':'voice head, first layer'}
# the attention DESIGN.md section 10 names as locked on one key
LOCKED_NAMED=('encoder_registry.encoders.motor.pool_attn','encoder_registry.encoders.motor.layers.0.attn',
    'encoder_registry.encoders.tactile.pool_attn','forward_model.blocks.0.cm_attn','forward_model.blocks.0.cmb_attn')
MEAS=REPO/'docs/gain_of_function_v3/measurements'


def _part(name):
    # a sense is named as the model's registry spells it (vestibular, proprioception, motor, ...), the
    # word the explainer uses too; SENSE_DEFINITIONS defines each where the card first names it
    sense_words = lambda s: s.replace('_',' ')
    block = re.fullmatch(r'motor_controller\.residual_blocks\.(\d+)\.0', name)
    if block:
        return f"motor controller, block {int(block.group(1)) + 1}"
    if name in PART_NAMES:
        return PART_NAMES[name]
    attention = re.fullmatch(r'forward_model\.blocks\.(\d+)\.(cm_attn|ct_attn|cmb_attn)', name)
    if attention:
        role = {'cm_attn':'attention between senses','ct_attn':'attention over time',
                'cmb_attn':'attention between senses and model steps'}[attention.group(2)]
        return f"forward model, block {int(attention.group(1)) + 1}: {role}"
    attention = re.fullmatch(r'prediction_heads\.heads\.([a-z_]+)\.attn', name)
    if attention:
        return f"{sense_words(attention.group(1))} prediction head's input selection"
    stage = re.fullmatch(r'prediction_heads\.heads\.([a-z_]+)\.mlp\.(\d+)', name)
    if stage:
        return f"{sense_words(stage.group(1))} prediction head, stage {int(stage.group(2)) + 1}"
    attention = re.fullmatch(r'encoder_registry\.encoders\.([a-z_]+)\.(?:layers|blocks)\.(\d+)\.attn', name)
    if attention:
        return f"{sense_words(attention.group(1))} encoder, attention layer {int(attention.group(2)) + 1}"
    attention = re.fullmatch(r'encoder_registry\.encoders\.([a-z_]+)\.(token_read|pool\.attn)', name)
    if attention:
        return f"{sense_words(attention.group(1))} encoder's input selection"
    # the code's own names (lem/model/efference.py EfferenceHeads.gaze_pool / body_pool, the voice head)
    named = {'gaze_pool':"gaze head's input selection", 'body_pool':"body head's input selection",
             'voice_head.attn':"voice head's input selection"}
    if name in named:
        return named[name]
    # a reconstruction decoder's measured layer is its first (lem_model.py recon_decoders: Linear, GELU,
    # Linear), for every sense alike
    decoder = re.fullmatch(r'recon_decoders\.([a-z_]+)\.0', name)
    if decoder:
        return f"the {sense_words(decoder.group(1))} reconstruction decoder"
    # the feed-forward network of a forward-model block or of an encoder layer (forward_model.py
    # ThreeStageBlock.ffn, encoders/cross_patch.py layers[n]["ffn"], encoders/swin_patch.py SwinBlock)
    ffn = re.fullmatch(r'forward_model\.blocks\.(\d+)\.ffn\.0', name)
    if ffn:
        return f"forward model, block {int(ffn.group(1)) + 1}: feed-forward network"
    ffn = re.fullmatch(r'encoder_registry\.encoders\.([a-z_]+)\.(?:layers|blocks)\.(\d+)\.ffn\.0', name)
    if ffn:
        return f"{sense_words(ffn.group(1))} encoder, feed-forward network of layer {int(ffn.group(2)) + 1}"
    # the per-patch layer each patch's reading enters first (encoders/cross_patch.py patch_mlp; the
    # motor encoder's patches are the body parts' rows)
    patch = re.fullmatch(r'encoder_registry\.encoders\.([a-z_]+)\.patch_mlp\.0', name)
    if patch:
        return (f"{sense_words(patch.group(1))} encoder's "
                + ("per-body-part" if patch.group(1) == "motor" else "per-patch") + " input layer")
    # encoders/mlp_project.py: each hidden layer is Linear, norm, activation, dropout, so mlp.0, mlp.4
    # and mlp.8 are the Linear layers of stages 1, 2 and 3
    mlp = re.fullmatch(r'encoder_registry\.encoders\.([a-z_]+)\.mlp\.(\d+)', name)
    if mlp and int(mlp.group(2)) % 4 == 0:
        return f"{sense_words(mlp.group(1))} encoder, stage {int(mlp.group(2)) // 4 + 1}"
    # lem/model/wellness.py: the grounded evaluators' networks exist only in the baseline wellness mode
    # and their outputs are no longer used (each term is its formula); kept so older checkpoints load
    grounded = re.fullmatch(r'wellness\.grounded\.([a-z_]+)\.net\.0', name)
    if grounded:
        return f"the grounded {sense_words(grounded.group(1))} evaluator's network"
    stage = re.fullmatch(r'(recon_decoders|encoder_registry\.encoders)\.([a-z_]+)\.(?:net\.)?(\d+)', name)
    if stage:
        sense = sense_words(stage.group(2))
        kind = 'reconstruction' if stage.group(1) == 'recon_decoders' else 'encoder'
        return f"{sense} {kind}, stage {int(stage.group(3)) + 1}"
    return PART_NAMES.get(name) or name


_HANDLE_WORDS={'upperarm':'upper arm','lowerarm':'forearm','lower_trunk':'lower trunk','gaze':'eyes (the gaze command)'}


def handle_words(name):
    """A body-layout handle in plain words: 'upperarm_L' -> 'left upper arm', 'gaze' -> 'eyes'."""
    name=str(name)
    side=''
    if name.endswith('_L'):side,name='left ',name[:-2]
    elif name.endswith('_R'):side,name='right ',name[:-2]
    return side+_HANDLE_WORDS.get(name,name.replace('_',' '))


def _cap(text):
    return text[:1].upper()+text[1:] if text else text


def _report(block):
    """Which report a liveness reading came from, in words: the module's name (lineage_series.LIVE_MODULES),
    never its record key ('contact_control')."""
    from .lineage_series import LIVE_MODULES
    if block == 'model':
        return "model's report"
    name = {blk: nm for blk, nm, _flag in LIVE_MODULES}.get(block, block).removeprefix('the ')
    return name + ("' report" if name.endswith('s') else "'s report")


def _response_name(block, name):
    """Name the measurement context, not just the layer shared by two checks."""
    part = name.removeprefix('efference.').removeprefix('lesson.')
    # the code's own names (lem/model/efference.py EfferenceHeads): body_mlp.0 and body_mlp.2 are the
    # body head's two hidden layers, gaze_in the gaze head's, vision_in the gaze copy's (vision_delta)
    heads = {'body_mlp.0': 'body head', 'body_mlp.2': 'body head, second layer', 'gaze_in': 'gaze head',
             'vision_in': 'gaze copy'}
    if part in heads:
        return heads[part] + (' — actual-command lessons' if block == 'efference' or '.lesson.' in name
                              else ' — all monitored inputs')
    # a module's own report names its layers without the module's prefix (review of 2026-09-25,
    # finding 8): lem/model/contact_head.py ContactHead.LIVE_LAYERS and motor_synergy.py
    # MotorSynergy.LIVE_LAYERS (S turns z into a posture change, E is the hindsight encoder)
    own = {'contact_control': {'ctx1': "the contact head's body-and-command input layer",
                               'ctx2': "the contact head's second context layer",
                               'patch': "the contact head's per-patch layer"},
           'synergy': {'S1': "the synergy map's S, layer 1", 'S2': "the synergy map's S, layer 2",
                       'E1': "the synergy map's hindsight encoder E, layer 1",
                       'E2': "the synergy map's hindsight encoder E, layer 2"}}
    if name in own.get(block, {}):
        return own[block][name]
    return _part(name)


# ONE LINE PER PART THE HEALTH CARD CAN NAME, printed where the card first names it (the operator's
# rule: the code's own word for a part, defined where it is first used)
PART_DEFINITIONS = (
    ('body head', 'The body head, one of the efference heads, predicts from the command a body was sent '
                  'how far each body part moves and turns over the next 50 ms and over the whole 250 ms command.'),
    ('gaze head', 'The gaze head, one of the efference heads, predicts from the eye command how far the '
                  'eyes turn over the next 50, 100 and 150 ms.'),
    ('gaze copy', 'The gaze copy, part of the efference heads, adds the eye command\'s expected effect to '
                  'the predicted next view.'),
    ('prediction head', 'A prediction head turns the forward model\'s state into one sense\'s prediction '
                        'of the next model step.'),
    ('contact head', 'The contact head predicts, for each skin patch, whether it will be touched in the '
                     'next 50 ms and whether by the body\'s own part or by an outside thing.'),
    ('voice head', 'The voice head outputs 80 voice-control numbers; no lesson trains it.'),
    # review of 2026-09-25, finding 8: each kind of layer the production record names, from the code
    ('wellness predictor', 'The wellness predictor (the code\'s ApproximateEvaluator) is the learned copy of wellness: it copies five of '
                              'the wellness terms from the tokens, so the planner can score imagined model steps.'),
    ("evaluator's network", "A wellness evaluator's network is a small learned stand-in for one wellness "
                            "term, kept so older checkpoints load; its output is no longer used, because "
                            "each term is computed by its formula."),
    ('tenure pose head', 'The tenure pose head predicts, from the model step a command began and the command '
                         'itself, how far each body part moves over the command\'s whole tenure (250 ms).'),
    ('tenure command projection', 'The tenure command projection turns a command into a vector that must pick '
                                'out its own tenure\'s change among recent decoys (the command-to-outcome '
                                'matching lesson).'),
    ('reconstruction decoder', 'A reconstruction decoder rebuilds one sense\'s present token from the '
                               'forward model\'s output, a check that the output still contains the '
                               'present model step.'),
    ('motor controller', 'The motor controller is the network that proposes a command for each body part; '
                         'its confidence head sets how much each of its residual blocks counts.'),
    ("persistent memory's outcome head", "The persistent memory's outcome head predicts, from the "
                                         "memory's state and the command, five measured needs of the body: "
                                         "pain, pleasant touch, physical capacity, skin pressure and skin "
                                         "shear."),
    ('feed-forward network', 'A feed-forward network is the two-layer part of a block that processes each '
                             'token on its own after the attention.'),
    ('per-patch input layer', 'An encoder\'s per-patch input layer is the first layer each patch\'s reading '
                              'passes through.'),
    ('per-body-part input layer', 'The motor encoder\'s per-body-part input layer is the first layer each '
                                  'body part\'s row of the command passes through.'),
    ('angular position encoding', 'The vision encoder\'s angular position encoding turns where each image '
                                  'patch looks, and from which eye, into numbers added to the patch.'),
    ("self-state's grounded dynamics", "The self-state's grounded dynamics predicts how the body's local "
                                       "functioning will change, from the observed present and the commands "
                                       "sent; its local layer is where each body site's reading meets the "
                                       "commands so far."),
    ('synergy map', 'The synergy map turns a few numbers of intention, z, into a whole-body posture change '
                    '(its part S) and learns from the changes the body achieved (its hindsight encoder E).'))


def _part_definitions(names):
    """The PART_DEFINITIONS lines for the parts these names mention, in table order, then one line per
    sense whose encoder, token, prediction head or reconstruction the names mention (the explainer's
    schema.SENSE_DEFINITIONS, so both pages define a sense in the same words)."""
    from mimo_bridge.explainer.schema import SENSE_DEFINITIONS
    text = ' '.join(str(n).lower() for n in names)
    senses = [s for s in SENSE_DEFINITIONS
              if re.search(r'\b%s (encoder|token|prediction head|reconstruction)' % s, text)]
    return ([line for key, line in PART_DEFINITIONS if key in text]
            + [f'{s.capitalize()} is {SENSE_DEFINITIONS[s]}.' for s in senses])


# MEASURED VERDICTS FOR FLAGGED LAYERS (operator, Sep 30 2026: the model alert read as gobbledygook). A layer
# whose idle units were switched off without any loss of accuracy is named here with that measurement; the
# hourly watcher treats the same layers as alerts only (docs/h84_launch_2026-09-18/hourly_watch.py
# LIVENESS_ALERT_ONLY). Any other flagged layer reads "not yet checked".
LAYER_VERDICTS = {
    # re-measured by docs/gain_of_function_v4/idle_units/idle_cost.py on ckpt_step_447350
    ('efference', 'body_mlp.0'): dict(
        width=256, idle=146,
        covered=192,
        what='forecasts of how far body parts move',
        found='its forecasts of how far body parts move still beat guessing "no movement" (missing by 0.69 cm, '
              'where the guess misses by 0.82 cm), and switching those units off changed that miss by less than '
              '0.0001 cm (measured at step 447,350 on 800 saved model steps of 8 bodies)',
        source='docs/gain_of_function_v4/idle_units/results/body_mlp.0__H84_fresh_drivers_20260918_seg41m_2030_resume_20260930_225714__ckpt_step_447350.json'),
    # re-measured by docs/gain_of_function_v4/idle_units/idle_cost.py on ckpt_step_447350
    ('efference', 'vision_in'): dict(
        width=128, idle=65,
        covered=65,
        what='predicted view',
        found='its predicted view is 34% closer to the real next picture than guessing "the picture stays the '
              'same", and switching those units off changed its error by under 1 in 10,000 of that lead (measured '
              'at step 447,350 on 800 saved model steps of 8 bodies)',
        source='docs/gain_of_function_v4/idle_units/results/vision_in__H84_fresh_drivers_20260918_seg41m_2030_resume_20260930_225714__ckpt_step_447350.json'),
}


def layer_verdict(blk, name, responding):
    """Whether a flagged layer's idle units were shown to cost nothing AT TODAY'S IDLE COUNT: the measurement covers
    up to `covered` units (the most of its least responsive units switched off with the change still under its
    line; Sep 30 2026), so more idle units than that need the test again."""
    known = LAYER_VERDICTS.get((blk, name))
    if not known:
        return 'Not yet checked whether the idle units cost any accuracy.'
    now = round((1 - responding) * known['width'])
    covered = max(known['idle'], known.get('covered', known['idle']))
    tested = f"With its {known['idle']} idle units of {known['width']} switched off, {known['found']}."
    if covered > known['idle']:
        tested += (f" Up to {covered} of its least responsive units can be switched off before the change reaches "
                   f"5 in 100 of that lead.")
    if now <= covered:
        return f"No action needed. {tested} About {now} are idle now."
    return (f"Needs a new check. {tested} About {now} are idle now, more than that test covered, so the test of "
            f"its {known['what']} must be repeated.")


def liveness_card(record, measurements=MEAS):
    """The Now tab's model health card (reviews/dashboards.md M10): every
    liveness report in the newest record (the top-level `liveness` block and any block's
    `liveness` / `pool_spread`, field contract 2026-09-23) against the 50-in-100 line of DESIGN.md
    section 10, and - until the model lane's permanent block exists - the parts the lead's sweep
    of ckpt_step_344400 found dead or locked, with the per-body-part readability of the motor
    token. Every sentence is built here; the page prints them as they are."""
    from .lineage_series import LIVE_LINE, RESPONDS, _liveness_parts
    from .plain_numbers import per, pct
    lines=[];parts=_liveness_parts(record or {})
    units=[p for p in parts if p[2]=='units'];pools=[p for p in parts if p[2]=='pool']
    below=sorted([p for p in units if p[3]<LIVE_LINE],key=lambda p:p[3])
    if units:
        title=(f"{len(below)} of {len(units)} layer response readings have under half their units responding" if below
               else f"All {len(units)} layer response readings have at least half their units responding")
        for blk,name,_k,v in below:
            lines.append(dict(kind='measure',text=f"In the {_report(blk)}, {_response_name(blk,name)}: {per(v,'units')} respond; the line is 50 of every 100."))
        if len(units)>len(below):
            lines.append(dict(kind='measure',text=f"{len(units)-len(below)} reported layer{'s are' if len(units)-len(below)>1 else ' is'} at or above the line."))
        # what the count means, never the raw input line (fix-readout-dashboards 2026-09-23)
        lines.append(dict(kind='note',text=f"What responding means: {RESPONDS}."))
    else:
        title='No layer in the newest record reports how many of its units respond'
    for blk,name,_k,v in sorted(pools,key=lambda p:p[3]):
        lines.append(dict(kind='measure',text=f"In the {_report(blk)}, {_part(name)}: attention spread {pct(v,2)} of an even spread (0% = it reads one input, 100% = every input evenly)."))
    if pools:
        lines.append(dict(kind='note',text='No line is drawn for attention spread: DESIGN.md section 10 names none.'))
    top=(record or {}).get('liveness')
    # where each shown reading comes from: the newest training record for the current readings, and
    # the dated step-344,400 files only for the lines that print them (they were pinned as the card's
    # only source, and the earlier-findings link opened the sweep beside the readability line)
    history=[]
    if isinstance(top,dict):
        if top.get('samples') is not None:
            lines.append(dict(kind='source',text=f"The model-wide report used {int(top['samples']):,} samples this window."))
        if top.get('revived'):
            lines.append(dict(kind='source',text='Revived by the repair: '+', '.join(_part(str(x)) for x in top['revived'])+'.'))
    else:
        sweep_path=measurements/'liveness_sweep_ckpt_step_344400.json'
        sweep=read_json_or(sweep_path,None)
        if isinstance(sweep,dict):
            history.append(dict(label='Liveness sweep of ckpt_step_344400 (2026-09-22)',path=sweep_path))
            lines.append(dict(kind='related',text=f"What responding means: {RESPONDS}."))
            lines.append(dict(kind='related',text="Until the model reports it (the repair lane's permanent liveness block), the parts the lead's sweep of ckpt_step_344400 found (2026-09-22, "
                              f"{sweep.get('moments','?')} lived model steps, CPU):"))
            for name,v in sorted((sweep.get('dead_layers') or {}).items(),key=lambda kv:kv[1]):
                lines.append(dict(kind='known',text=_cap(f"{_part(name)}: {per(v,'units')} respond.")))
            att=sweep.get('attention') or {}
            for name in LOCKED_NAMED:
                if name in att:
                    lines.append(dict(kind='known',text=_cap(f"{_part(name)}: attention spread {pct(att[name].get('entropy_share_of_uniform',0),2)} of an even spread over its {att[name].get('keys')} inputs (locked on one, DESIGN.md section 10).")))
    ph_path=measurements/'per_handle_readability_ckpt_step_344400.json'
    ph=read_json_or(ph_path,None)
    if isinstance(ph,dict) and ph.get('largest_gap'):
        g=ph['largest_gap'];names=g.get('handles_below') or []
        measured=local_stamp(ph.get('measured'))
        measured=clock(measured) if measured else (ph.get('measured') or 'on an unrecorded date')
        lines.append(dict(kind='known',text=f"The model reads {len(names)} of {len(ph.get('per_handle_share') or {})} body parts' commands at the level of arithmetic rounding: turning one of them 20 degrees moves the motor token (the vector through which the model reads its own command) by under "
                          f"{per(ph['line'],'parts of its size')}, the measured line; the next part moves it {per(g['between'][1],'parts of its size')} (measured {measured} on ckpt_step_344400). Unread: "+', '.join(handle_words(n) for n in names)+'.'))
        history.append(dict(label=f'Per-body-part readability of ckpt_step_344400 (measured {measured})',path=ph_path))
    watching = (top or {}).get('watching', {}) if isinstance(top, dict) else {}
    revived = (top or {}).get('revived', []) if isinstance(top, dict) else []
    # --liveness-rule none (lem/model/liveness.py): no unit is revived, so no repair check ever comes
    revival_off = isinstance(top, dict) and top.get('revive_rule') == 'none'
    issues = [dict(name=_cap(_response_name(blk,name)), source=blk, responding=round(100 * v),
                   definition=' '.join(_part_definitions([_response_name(blk, name)])) or None,
                   reading=('Just under 50' if round(100 * v) == 50 else str(round(100 * v))),
                   verdict=layer_verdict(blk, name, v),
                   verdict_source=_repo_url(REPO / LAYER_VERDICTS[(blk, name)]['source'])
                   if (blk, name) in LAYER_VERDICTS else None,
                   repair=('Revival is off (--liveness-rule none): reported only; units revived before keep their '
                           'routing and learning.'
                           if revival_off and name in watching else
                           'Collecting fresh observations after loading; the first repair check is still pending.'
                           if (watching.get(name) or {}).get('ready') is False else
                           'Automatic repair is monitoring this layer.' if name in watching else
                           'Monitored only; automatic repair does not cover this layer.'))
              for blk, name, _kind, v in below]
    repaired_ok = sum(1 for name in revived
                      if any(blk == 'model' and part == name and v >= LIVE_LINE
                             for blk, part, _kind, v in units))
    summary = (f"{len(below)} response reading{'s need' if len(below) != 1 else ' needs'} attention" if below else
               'No layer readings below the warning line' if units else 'No current layer readings')
    return dict(title=title,lines=lines,reports=len(parts),below=len(below),
                summary=summary,status='attention' if below else 'ok' if units else 'unmeasured',
                issues=issues,reported_layers=len(units),
                readings=[dict(name=_cap(_response_name(blk,name)),source=blk,
                               responding=round(100*v,1),below=v<LIVE_LINE)
                          for blk,name,_kind,v in units],
                scope_explanation=('Actual-command lessons measure responses while learning from the command that really moved the body. '
                                   'All monitored inputs also include deliberately shuffled commands used for comparison. '
                                   'These are separate readings of the same layer: a healthy comparison reading cannot clear a weak lesson reading.'),
                attention=[dict(name=_cap(_part(name)),spread=pct(v,2)) for _blk,name,_kind,v in sorted(pools,key=lambda p:p[3])],
                # one plain sentence on the card; every definition sits behind a "Definitions" link
                # (operator, Sep 30 2026: the card printed 642 words defining all 59 watched layers)
                explanation=('These are parts of the model, not body parts. A part is flagged when fewer than half '
                             'of its units (its small adjustable pieces) react to changing inputs.'),
                glossary=_part_definitions([i['name'] for i in issues]
                                           + [_response_name(b, n) for b, n, _k, _v in units]
                                           + [_part(n) for _b, n, _k, _v in pools]),
                repair_summary=(f"{repaired_ok} of {len(revived)} previously repaired layers are now above the warning line."
                                if revived else None),
                source_text=_record_source(record) if units or pools else None,
                history_sources=[dict(label=h['label'],url=_repo_url(h['path'])) for h in history if _repo_url(h['path'])],
                # the earlier measurement, only when the card prints the sweep itself
                source=next((_repo_url(h['path']) for h in history if 'sweep' in h['path'].name),None))


def _repo_url(path):
    """A /files/ link to a file inside the repository, or None (a test's own folder, a missing file)."""
    try:
        path=Path(path).resolve()
        return file_url(REPO,path) if path.is_file() and path.is_relative_to(REPO.resolve()) else None
    except (OSError,ValueError):
        return None


def _record_source(record):
    """Which training record the current readings come from, with its step and AM/PM time."""
    record=record or {}
    step,stamp=record.get('step'),record.get('timestamp')
    if not isinstance(step,(int,float)):
        return 'Readings from the newest training record.'
    return (f"Readings from the newest training record: lineage step {int(step):,}"
            + (f", written {clock(stamp)}." if isinstance(stamp,(int,float)) else '.'))


def now_summary(record, args, panel):
    from .training_brief import assessment
    return assessment(record or {}, args or {}, panel)


def observation_cards(training, record, live):
    """The "What the bodies are doing now" cards, every sentence built here (section 11)."""
    from .plain_numbers import fixed, pct, per
    from .lineage_series import SERIES, say
    cards=[]
    rolls=posture_view(record or {})
    if rolls:
        first=next((r for r in rolls if r['body']==1),None);most=max(rolls,key=lambda r:r['roll'])
        cards.append(dict(title='How the bodies lie',value=(f"Body 1: {fixed(first['roll'],2)} degrees from lying on its back" if first else 'Body 1: no valid reading'),
            notes=['0 degrees = on its back, 90 = on its side, 180 = belly down.',
                   f"The most-turned of the {len(rolls)} bodies is body {most['body']}, at {fixed(most['roll'],2)} degrees.",
                   'These are separate worlds with different movement histories. This pose reading cannot tell whether a body turned deliberately; intentional rolling has not been established.']))
    else:
        cards.append(dict(title='How the bodies lie',value='Unmeasured',notes=['This record carries no body_orientation reading; a missing reading is not lying on the back.']))
    # every body's collapse watch against its sense's line (mimo_bridge/deaf_senses.py, 2026-09-25): the
    # hourly watcher's DEAF SENSE ALERT names the same bodies from an hour's mean
    try:
        from mimo_bridge import deaf_senses
        per_body=(record or {}).get('token_temporal_std_bodies')
        cards.append(deaf_senses.card(per_body,_record_source(record) if per_body else None))
    except Exception as exc:   # noqa: BLE001 - one card's failure must not cost the other cards
        cards.append(dict(title='Deaf senses',value='Unavailable',notes=[f'The per-body reading could not be computed ({type(exc).__name__}).']))
    st=(live or {}).get('self_touch') or {}
    pairs=st.get('pairs')
    if isinstance(pairs,(int,float)):
        notes=[f"A pair means two body parts touching. This is the average number of such pairs touching at once per body, across the {len(rolls) or 'reported'} bodies in the window (0 = no self-touch recorded in this reporting interval)."]
        imp=st.get('predicted_better_than_shuffled')
        if isinstance(imp,(int,float)):
            notes.append("The prediction comparison scores all tactile information for bodies touching themselves, including contact with the floor. It does not test deliberate self-touch.")
        else:
            notes.append('The prediction of self-touch is not scored in this record.')
        cards.append(dict(title='Touching itself',value=say(SERIES['self_pairs'],pairs),notes=notes))
    else:
        cards.append(dict(title='Touching itself',value='Unmeasured',notes=['This record carries no self-contact count.']))
    c=((training or {}).get('caregiver') or [None])[0]
    if isinstance(c,dict) and c.get('mode')!='off':
        here=c.get('adult_present') is True or (c.get('adult_visibility') or 0)>0
        phases={'starting':'The adult is starting an interaction.','presenting':'The adult is bringing objects closer.',
                'engaged':'Objects are within reach.','withdrawing':'The adult is taking objects away.','absent':'The objects are away.'}
        dep=c.get('adult_departures')
        notes=[(f"The adult has left {int(dep):,} time{'s' if int(dep)!=1 else ''} this episode. " if isinstance(dep,(int,float)) else '')+phases.get(c.get('phase'),'Where the objects are is not recorded.')]
        d=c.get('head_horizontal_distance_m')
        notes.append(f"The adult's head is {fixed(100*d,2)} cm from the baby's eyes, measured horizontally." if isinstance(d,(int,float)) else "Where the adult's head is, is not recorded.")
        cards.append(dict(title="The adult in body 1's world",value='Present' if here else 'Away',notes=notes))
    else:
        cards.append(dict(title="The adult in body 1's world",value='Not recorded',notes=['The record carries no caregiver reading; a missing reading is not an absent adult.']))
    return cards


def ancestry_bounds(root, training):
    """Follow resumes, with a step bound at each fork; never sweep all runs. For the experiment view the mirror comes
    first, then the ancestry of its nearest ancestor checkpoint inside the project (the mirror's own args.json resumes a
    checkpoint outside the project, which is not followed)."""
    bounds = {}
    if not training.get('run'):
        return bounds
    if training.get('view') == 'experiment':
        bounds[str(confined(root, training['run']))] = training.get('step')
        parent = (training.get('experiment') or {}).get('project_parent') or {}
        if parent.get('run') and isinstance(parent.get('step'), int):
            try:
                for run, limit in ancestry_bounds(root, dict(run=parent['run'], step=parent['step'])).items():
                    bounds.setdefault(run, limit)
            except ValueError:
                pass
        return bounds
    run = confined(root, training['run'])
    limit = training.get('step')
    for _ in range(500):
        if str(run) in bounds:
            break
        bounds[str(run)] = limit
        try:
            args = read_json(run/'args.json')
            resume = args.get('resume')
            if not resume:
                break
            checkpoint = checkpoint_reference(root, resume)
            # An expanded checkpoint can be stored in the new run itself. Follow
            # its explicit migration parents before applying the resume boundary;
            # otherwise that self-reference hides all earlier earned evidence.
            seen_checkpoints=set()
            for _migration in range(500):
                if str(checkpoint) in seen_checkpoints:
                    raise ValueError('cyclic checkpoint migration')
                seen_checkpoints.add(str(checkpoint))
                parent=None
                for name in ('continuity_migration.json','context_growth.json','trunk_migration.json'):
                    if not (checkpoint/name).exists():
                        continue
                    note=read_json(checkpoint/name)
                    if note.get('checkpoint') and checkpoint_reference(root,note['checkpoint'])==checkpoint:
                        parent=note.get('parent')
                        break
                if not parent:
                    break
                checkpoint=checkpoint_reference(root,parent)
            match = re.fullmatch(r'ckpt_step_(\d+)(?:_.*)?', checkpoint.name)
            if not match or checkpoint.parent.name != 'checkpoints':
                break
            checkpoint_run=checkpoint.parent.parent
            saved_config=read_json(checkpoint_run/'args.json') if (checkpoint_run/'args.json').is_file() else {}
            run=confined(root,saved_config['out']) if saved_config.get('checkpoint_storage') and saved_config.get('out') else checkpoint_run
            limit=int(match[1])
        except (OSError, ValueError, TypeError):
            break
    return bounds


def lineage_relation(root, subject, bounds):
    try:
        path = checkpoint_reference(root, subject)
        match = re.fullmatch(r'ckpt_step_(\d+)(?:_.*)?', path.name)
        checkpoint_run=path.parent.parent
        saved_config=read_json(checkpoint_run/'args.json') if (checkpoint_run/'args.json').is_file() else {}
        if saved_config.get('checkpoint_storage') and saved_config.get('out'):
            checkpoint_run=confined(root,saved_config['out'])
        limit = bounds.get(str(checkpoint_run))
        if match and limit is not None and int(match[1]) <= limit:
            return 'ancestor_checkpoint'
    except (OSError,ValueError, TypeError):
        pass
    return 'not_linked_to_current_model'


def evidence_records(root, registry, bounds):
    records = []
    for original in registry['records']:
        record = dict(original)
        record['valid'] = False
        record['sources'] = []
        try:
            receipts = record['receipts']
            if not receipts:
                raise ValueError('No source receipt')
            payloads = []
            for receipt in receipts:
                path = confined(root, receipt['path'])
                if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['sha256']:
                    raise ValueError('The source changed after review')
                payloads.append(read_json(path))
                record['sources'].append(dict(label=path.name,url=file_url(root,path)))
            if not record['checks']:
                raise ValueError('No reviewed result check')
            for check in record['checks']:
                value = payloads[check['receipt']]
                for key in check['path'].split('.'):
                    value = value[key]
                if type(value) is not type(check['equals']) or value != check['equals']:
                    raise ValueError('The receipt no longer matches the assessment')
            for link in record.get('links', []):
                path = confined(root, link)
                if path.is_file():
                    record['sources'].append(dict(label=path.name,url=file_url(root,path)))
            if record['id'] == 'wellness-choice':
                report = payloads[0]
                record['comparison'] = report['B']
                record['worlds'] = len(report['worlds_answering'])
            record['valid'] = True
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            record['review_issue'] = str(exc)
        record['lineage'] = lineage_relation(root, record['subject'], bounds)
        # Even an ancestor's live-behavior test does not establish current retention.
        record['current_live_ability_proven'] = False
        records.append(record)
    return records


def reviewed_behavior_records(root,bounds):
    """Attach negative native observations without awarding acquisition or retention."""
    from .experiment_board import registry as work_registry,directory
    mappings={'action_forecast':('predict','Its native movement-effects forecasts'),
        'toy':('correct','Looking when the toy moves or disappears'),
        'body_floor':('roll','Whole-body turning on the floor'),
        'free_objects':('contact','Reaching and grasping a free object'),
        'support':('support','Using a support to raise its body')}
    records=[]
    for job in work_registry(root).get('jobs',[]):
        if not job['id'].startswith('development-v3-') or not job.get('run_dir'):continue
        folder=directory(root,job)
        try:
            status=read_json(folder/'status.json')
            if status.get('stage')!='complete':continue
        except (OSError,ValueError):continue
        record=dict(id=job['id'],kind='live_behavior',outcome='observed_limit',valid=False,
            sources=[],capability=None,current_live_ability_proven=False,lineage='not_linked_to_current_model')
        try:
            reviewed=bool(job.get('review'))
            if reviewed:
                review_path=confined(root,job['review']);review=read_json(review_path)
            else:
                declaration=read_json(folder/'declaration.json')
                if not declaration.get('checkpoint_sha256'):
                    raise ValueError('Older observation needs review: its declaration did not bind model weights')
                from .review_completed_developmental import review as summarize_observations
                review=summarize_observations(folder,job['id'])
                review['reviewed_at']=status['timestamp']
            capability,title=mappings[review['profile']]
            record.update(capability=capability,title=title,subject=review['model_identity']['checkpoint'],
                date=datetime.fromtimestamp(review['reviewed_at'],timezone.utc).date().isoformat(),
                reviewed_at=review['reviewed_at'])
            record['lineage']=lineage_relation(root,record['subject'],bounds)
            for name,key in (('report.json','report_sha256'),('declaration.json','declaration_sha256')):
                if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=review[key]:
                    raise ValueError('Native test report or declaration changed after review')
            declaration=read_json(folder/'declaration.json');report=read_json(folder/'report.json')
            model=review['model_identity'];checkpoint=checkpoint_reference(root,model['checkpoint'])
            if checkpoint!=checkpoint_reference(root,declaration['checkpoint']):raise ValueError('Reviewed model differs from the test subject')
            from .evidence_identity import checkpoint_identity
            identity=checkpoint_identity(checkpoint,model['model_sha256'])
            if declaration.get('checkpoint_sha256') and declaration['checkpoint_sha256']!=model['model_sha256']:
                raise ValueError('Model identity differs from the original declaration')
            if review.get('capability_pass') is not False or not review.get('complete_replay_verified'):
                raise ValueError('Observation review cannot grant an acquired ability')
            worlds=report['results']
            if not worlds or not all(w.get('same_action_replay_exact') and
                (w.get('complete_model_replay_exact') or w.get('complete_model_replay',{}).get('complete_model_replay_exact')) for w in worlds):
                raise ValueError('Complete model and physical replay evidence is missing')
            capability,title=mappings[review['profile']]
            record.update(valid=True,capability=capability,title=title,summary=review['summary'],
                date=datetime.fromtimestamp(review['reviewed_at'],timezone.utc).date().isoformat(),
                subject=str(checkpoint),missing=review['limitation'],method=review['decision']+' '+review['limitation'],
                reviewed_at=review['reviewed_at'],model_identity=dict(model,**identity),
                validation='reviewed_observation' if reviewed else 'automated_validated_observation',
                sources=([dict(label='Reviewed observations',url=file_url(root,review_path))] if reviewed else [])+
                    [dict(label='Original report',url='/experiments/'+quote(job['id'],safe='')+'/report.json'),
                     dict(label='Declared test and model identity',url='/experiments/'+quote(job['id'],safe='')+'/declaration.json')])
            if identity.get('availability')=='retired':
                record['missing']+=' '+identity['limitation']
                record['method']+=' Source snapshot retired under the recorded storage policy; historical identity matches its completed retirement receipt.'
            record['lineage']=lineage_relation(root,str(checkpoint),bounds)
        except (OSError,ValueError,KeyError,TypeError) as error:
            record.update(summary='Saved observations need review.',review_issue=str(error),missing=str(error),method='')
            record.setdefault('title',job['title']);record.setdefault('subject',job.get('run_dir',''));record.setdefault('date','')
        records.append(record)
    return records


def legacy_state(root,now):
    saved=root/'Saved/mimo_bridge';out={}
    for key,name in (('dashboard','dashboard_legacy.html'),('history','lifetime.html')):
        p=saved/name
        out[key]=dict(updated_at=p.stat().st_mtime,age_seconds=max(0,now-p.stat().st_mtime)) if p.exists() else None
    try:out['updater']=read_json(saved/'legacy_dashboard_service.json')
    except (OSError,ValueError):out['updater']=None
    return out


HISTORY_KEEP = 48


def refresh_legacy(root=REPO):
    """One owner updates both preserved diagnostic pages independently of requests."""
    from mimo_bridge.tools import dashboard
    saved=root/'Saved/mimo_bridge'
    try:
        from mimo_bridge.tools import lifetime
        lifetime.build(max_age_s=1800)  # Surface a history failure to the service status.
        dashboard.build()
        history=saved/'dashboard_history'
        history.mkdir(parents=True,exist_ok=True)
        archive=history/time.strftime('dashboard_%Y-%m-%d_%H00.html')
        if not archive.exists():
            shutil.copy2(saved/'dashboard_legacy.html',archive)
        # Keep the newest 48 hourly copies, as dashboard.main always did; this loop kept all
        # of them (204 files, 27 MB on 2026-09-22).
        for old in sorted(history.glob('dashboard_*.html'))[:-HISTORY_KEEP]:
            try:old.unlink()
            except OSError:pass
        status=dict(updated_at=time.time(),status='updated',error=None)
    except Exception as exc:
        status=dict(updated_at=time.time(),status='failed',error=str(exc))
    atomic_write(saved/'legacy_dashboard_service.json',json.dumps(status,indent=2))
    return status


def legacy_loop(stop):
    from mimo_bridge.tools.training_continuation import OwnerLock
    try:lock=OwnerLock(SAVED/'legacy_dashboard.lock')
    except OSError:return  # An explicitly launched old updater can own the same pages.
    try:
        while not stop.is_set():
            refresh_legacy()
            stop.wait(30)
    finally:lock.close()


_MP4_CACHE = {}


def complete_mp4(path):
    """Ignore video files whose MP4 container has not finished writing."""
    try:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key in _MP4_CACHE:
            return _MP4_CACHE[key]
        types = set()
        with path.open('rb') as stream:
            offset = 0
            while offset < stat.st_size:
                head = stream.read(8)
                if len(head) != 8:
                    return False
                size, kind = struct.unpack('!I4s', head)
                header_size = 8
                if size == 1:
                    ext = stream.read(8)
                    if len(ext) != 8:
                        return False
                    size, header_size = struct.unpack('!Q', ext)[0], 16
                if size == 0:
                    size = stat.st_size-offset
                if size < header_size or offset+size > stat.st_size:
                    return False
                types.add(kind)
                offset += size
                stream.seek(offset)
        complete = {b'ftyp',b'moov',b'mdat'}.issubset(types)
        if len(_MP4_CACHE) > 500:
            _MP4_CACHE.clear()
        _MP4_CACHE[key] = complete
        return complete
    except OSError:
        return False


def _video_control_annotated(outside):
    control_path=outside.with_name(outside.stem+'_control.json')
    try:
        control=read_json(control_path)
        if not isinstance(control,dict):control={}
    except (OSError,ValueError):
        control={}
    return (control.get('version')=='body_control_video_v1'
            and control.get('observer_overlay') is True and bool(control.get('frames')))


EXPERIMENT_MIRRORS = 'Saved/mimo_bridge/experiment_mirror'
PRODUCTION_CLIPS_MAX = 12        # production's ancestry clips, as before the merge
VIDEO_LIST_MAX = 36              # the whole list: production's clips keep their share, experiments fill the rest


def experiment_runs(root):
    """{mirror folder: 'live' | 'ended' | 'not confirmed'} of every experiment run copied into
    Saved/mimo_bridge/experiment_mirror (tools/experiment_mirror.py). Left out: folders whose name starts with '_'
    (runs moved aside) and any folder that resolves outside the mirrors folder (a link). live = the experiment view
    shows it now and its owner records phase "training" with the trainer process alive (experiment_mirror.live_view);
    ended = its mirror's record says the owner state showed the end, or the run was copied once after its end; not
    confirmed = neither."""
    from .experiment_mirror import live_view, STATUS
    base = Path(root)/EXPERIMENT_MIRRORS
    if not base.is_dir():
        return {}
    view = live_view(root)
    out = {}
    for folder in sorted(base.iterdir()):
        try:
            if (not folder.is_dir() or folder.name.startswith('_') or not (folder/'videos').is_dir()
                    or not folder.resolve().is_relative_to(base.resolve())):
                continue
        except OSError:
            continue
        status = read_json_or(folder/STATUS, {}) or {}
        training = bool(view and view['run_name'] == folder.name and view.get('phase') == 'training'
                        and view.get('trainer_alive'))
        out[folder] = ('live' if training else
                       'ended' if isinstance(status, dict) and (status.get('ended') or status.get('copied_once'))
                       else 'not confirmed')
    return out


def experiment_clips(root, folder, state, active_run):
    """The complete clip pairs (step_* and moment_* event clips) of one mirror folder, as list entries."""
    out = []
    for outside in sorted((folder/'videos').glob('*.mp4')):
        match = re.fullmatch(r'(?:moment_(?P<event>[A-Za-z0-9]+)_)?step_(?P<step>\d+)(?:_\d+)?', outside.stem)
        if not match:
            continue
        eyes = outside.with_name(outside.stem+'_eye.mp4')
        if not complete_mp4(outside) or not complete_mp4(eyes):
            continue
        event = match['event']
        out.append(dict(id=outside.relative_to(root).as_posix(),step=int(match['step']),
            recorded_at=outside.stat().st_mtime,run_name=folder.name,
            previous_run=folder.relative_to(root).as_posix()!=active_run,
            outside=file_url(root,outside),eyes=file_url(root,eyes),experiment=True,experiment_state=state,
            kind='event' if event else 'observation',events=[event] if event else [],outcome=None,
            control_annotated=_video_control_annotated(outside)))
    return out


def videos_for(root, bounds, active_run):
    """THE RECORDINGS LIST, newest first by recording time (the clip file's modification time): the current model's
    ancestry in the project (production's runs, at most PRODUCTION_CLIPS_MAX clips from its first 3 runs, a restored
    ancestor's clips only up to its fork step, step_* clips only, as before) merged with the newest clips of every
    experiment run copied into Saved/mimo_bridge/experiment_mirror (step_* clips and moment_* event clips, live or
    ended) up to VIDEO_LIST_MAX entries in all, so production's clips keep their share (operator, Oct 13, 2026: "why is
    the dashboard showing old video evidence and not the most recent!"). Each entry names its run and model step; an
    experiment entry says whether that run is live or ended. A mirror folder that cannot be read is left out. Every
    file served is inside the project (file_url, confined)."""
    pairs = []
    mirrors = (Path(root)/EXPERIMENT_MIRRORS).resolve()
    production = [run for run in bounds if not Path(run).resolve().is_relative_to(mirrors)][:3]
    for run in production:
        directory = Path(run)/'videos'
        candidates = sorted(directory.glob('step_*.mp4'), key=lambda p:p.name, reverse=True)
        for outside in candidates:
            if outside.stem.endswith('_eye'):
                continue
            eyes = outside.with_name(outside.stem+'_eye.mp4')
            if not complete_mp4(outside) or not complete_mp4(eyes):
                continue
            match = re.match(r'step_(\d+)',outside.stem)
            step = int(match[1]) if match else None
            # A restored ancestor must not contribute footage from its discarded future.
            limit = bounds[run]
            if step is None or limit is None or step > limit:
                continue
            pairs.append(dict(id=outside.relative_to(root).as_posix(),step=step,
                recorded_at=outside.stat().st_mtime,run_name=Path(run).name,
                previous_run=Path(run).relative_to(root).as_posix()!=active_run,
                outside=file_url(root,outside),eyes=file_url(root,eyes),experiment=False,experiment_state=None,
                kind='observation',events=[],outcome=None,control_annotated=_video_control_annotated(outside)))
            if len(pairs) >= PRODUCTION_CLIPS_MAX:
                break
        if len(pairs) >= PRODUCTION_CLIPS_MAX:
            break
    clips = []
    for folder, state in experiment_runs(root).items():
        try:
            clips += experiment_clips(root, folder, state, active_run)
        except (OSError, ValueError) as exc:          # a folder moved aside mid-read, a link: that folder only
            print(f'Dashboard: the clips of {folder.name} could not be read: {type(exc).__name__}: {exc}', flush=True)
    clips.sort(key=lambda v:v['recorded_at'], reverse=True)
    pairs += clips[:max(0, VIDEO_LIST_MAX-len(pairs))]
    pairs.sort(key=lambda v:v['recorded_at'], reverse=True)
    return pairs


def skin_regions_view(row):
    """Where the bodies are being touched, with the body and the time scope named.

    Per-body window means come from `skin_regions_per_body` when the trainer writes it
    (a list, one {region: {patches, self_patches, force_share}} per body). Until then
    `skin_regions` is ONE body - the last one in the trainer's list - at the window's last
    step (train_mimo.py, the self-contact accumulator), and is labelled as exactly that."""
    per_body = row.get('skin_regions_per_body')
    if isinstance(per_body, list) and per_body:
        bodies = []
        for index, regions in enumerate(per_body):
            if isinstance(regions, dict):
                touched = {name: value for name, value in regions.items()
                           if isinstance(value, dict) and (value.get('patches') or 0) > 0}
                bodies.append(dict(body=index + 1, regions=touched))
        return dict(scope='window means per body', bodies=bodies, body_count=len(per_body))
    single = row.get('skin_regions')
    if isinstance(single, dict):
        count = row.get('n_envs') or len(row.get('body_orientation') or []) or None
        touched = {name: value for name, value in single.items()
                   if isinstance(value, dict) and (value.get('patches') or 0) > 0}
        return dict(scope='one body at the last model step of the window', body_count=count,
                    bodies=[dict(body=count, regions=touched, last_in_list=True)])
    return dict(scope=None, bodies=[], body_count=None)


def posture_view(row):
    """Each body's roll from lying on its back (0 = on its back, 90 = on its side, 180 = belly
    down), from body_orientation. rolled_over_fraction is not used: it counts steps whose
    episode maximum ever passed 120 degrees, and that maximum includes turning around the pen."""
    out = []
    for index, body in enumerate(row.get('body_orientation') or []):
        if isinstance(body, dict) and body.get('valid') and isinstance(body.get('roll_degrees'), (int, float)):
            out.append(dict(body=index + 1, roll=min(180.0, abs(float(body['roll_degrees']))),
                            episode_max=body.get('maximum_roll_degrees')))
    return out


def _fresh_line(every, steps_per_second):
    """Three live-view periods in seconds (a picture older than that means the run stopped
    publishing), or None when the period is unknown."""
    try:
        every = int(every or 0)
    except (TypeError, ValueError):
        every = 0
    if every > 0 and isinstance(steps_per_second, (int, float)) and steps_per_second > 0:
        return 3.0 * every / float(steps_per_second)
    return None


# when the live-view period is unknown (no --live-view-every in the launch, or no step rate), a
# picture counts as current under the verdict's own line for a live training record: 10 minutes
LIVE_FALLBACK_S = 600


def live_texts(out, row, args):
    """The live tab's sentences, built here (section 11): publishing only when the newest picture
    is younger than three live-view periods (reviews/dashboards.md m2)."""
    from .plain_numbers import fixed
    from .lineage_series import SERIES, say
    sps = (row or {}).get("model_steps_per_s")
    fa, sa = out.get("frame_age_seconds"), out.get("state_age_seconds")
    fl = _fresh_line((args or {}).get("live_view_every"), sps)
    sl = _fresh_line((args or {}).get("live_state_every"), sps)
    if fl is not None and out.get("view") == "experiment":
        # the experiment view's picture is the mirror's copy, up to one copy period older than the run's picture
        from .experiment_mirror import TICK_S
        fl += TICK_S
    fresh_frame = fa is not None and fa <= (fl if fl is not None else LIVE_FALLBACK_S)
    fresh_state = sa is not None and sa <= (sl if sl is not None else LIVE_FALLBACK_S)
    out.update(frame_publishing=bool(fresh_frame),state_publishing=bool(fresh_state),
               frame_fresh_seconds=fl if fl is not None else LIVE_FALLBACK_S,
               state_fresh_seconds=sl if sl is not None else LIVE_FALLBACK_S,
               observed_at=time.time())
    out["publishing"] = bool(fresh_frame or fresh_state)
    line_words = (f"three live-view periods{' and the mirror copy period' if out.get('view') == 'experiment' else ''}, "
                  f"{_age_words(fl)}" if fl is not None
                  else f"{_age_words(LIVE_FALLBACK_S)}, the live-record line (the live-view period is unknown)")
    if fresh_frame and out.get("view") == "experiment":
        from .experiment_mirror import TICK_S
        pub = (f"yes, from the experiment run {out.get('run_name')}, copied by the experiment mirror, which looks for a "
               f"new picture every {TICK_S:g} second{'' if TICK_S == 1 else 's'}")
    elif fresh_frame:
        pub = f"yes, from {out.get('run_name') or 'the current run'}"
    elif fa is not None:
        pub = (f"no: the last picture is {_age_words(fa)} old, over {line_words}; "
               f"the run is not publishing")
    else:
        pub = "no: this run has never published a picture"
    st = out.get("self_touch") or {}
    t = {"pub": pub, "fresh": "no picture" if fa is None else f"{_age_words(fa)} ago",
         "age": "no picture" if fa is None else f"picture {_age_words(fa)} old"}
    pairs = st.get("pairs")
    t["selftouch"] = ("unmeasured" if not isinstance(pairs, (int, float)) else
                      say(SERIES["self_pairs"], pairs) + " (pairs of its own body parts touching at an average model step; 0 = none recorded in this reporting interval)"
                      + ("" if not isinstance(st.get("force_share"), (int, float)) else
                         f"; {fixed(100 * st['force_share'], 2)}% of the touch force"))
    imp = st.get("predicted_better_than_shuffled")
    t["selfpred"] = ("No tactile prediction comparison in this record" if not isinstance(imp, (int, float)) else
                     f"knowing its own command improves the prediction by {say(SERIES['self_touch_act'], imp)}, "
                     "against another body's command (below 0% = the real command did worse). "
                     "This scores all tactile information for bodies touching themselves, including floor contact; it does not isolate self-touch.")
    rolls = posture_view(row or {})
    t['touch_summary'] = ('No touch count available' if not isinstance(pairs,(int,float)) else
                          f"{say(SERIES['self_pairs'], pairs)}: body parts touching each other, averaged across {len(rolls) or 'the recorded'} bodies.")
    t["posture"] = ("unmeasured" if not rolls else
                    ", ".join(f"body {r['body']} {fixed(r['roll'], 2)} degrees" for r in rolls)
                    + " (0 degrees = on its back, 90 = on its side, 180 = belly down)")
    view = st.get("regions_view") or {}

    def patches(n):
        n = float(n or 0)
        return (f"{fixed(n, 2)} patch" + ("es" if n != 1 else "")) if n >= 1 or n == 0 else \
            f"{fixed(100 * n, 2)} patches per 100 model steps"

    def describe(regions):
        return " · ".join(f"{name} ({patches(v.get('patches'))}"
                          + (f", {patches(v.get('self_patches'))} by its own body" if v.get("self_patches") else "") + ")"
                          for name, v in (regions or {}).items()) or "nothing"
    bodies = view.get("bodies") or []
    if bodies and view.get("scope") == "window means per body":
        t["regions"] = " | ".join(f"Body {b['body']} (window mean): {describe(b['regions'])}" for b in bodies)
    elif bodies:
        t["regions"] = (f"Touched, body {bodies[0].get('body') or '?'} of {view.get('body_count') or '?'} only (the last "
                        f"body the trainer lists), at the last model step of the window: {describe(bodies[0]['regions'])}. "
                        f"The trainer does not yet write per-body window means per region.")
    else:
        t["regions"] = ""
    out["texts"] = t
    return out


def live_state(root, training, row=None, args=None):
    """What the live tabs need: whether the run is publishing, how old the newest picture
    is, and whether a recording is going. Pictures are NOT collected here - the run
    overwrites one file - so this is three stat() calls."""
    run = training.get("run")
    if not run:
        return {"publishing": False, "why": "no run is training",
                "texts": {"pub": "no: no run is training", "fresh": "no picture", "age": "no picture",
                          "selftouch": "unmeasured", "selfpred": "no self-touch scored yet",
                          "posture": "unmeasured", "regions": ""}}
    live = Path(root) / run / "live"
    frame, state = live / "frame.jpg", live / "state.npz"
    now = time.time()
    out = {"run": run, "run_name": training.get("run_name"), "view": training.get("view") or "production",
           "frame": file_url(Path(root), frame) if frame.exists() else None,
           "frame_age_seconds": (now - frame.stat().st_mtime) if frame.exists() else None,
           "state_age_seconds": (now - state.stat().st_mtime) if state.exists() else None,
           "sensors_published": (live / "sensors.npz").exists(),
           # copy-and-paste for a PowerShell prompt at the repo root: the package lives
           # under Source/Python and the venv has mujoco
           "viewer_command": ('$env:PYTHONPATH="Source\\Python"; '
                              '.venv_mimo\\Scripts\\python.exe -m mimo_bridge.tools.live_viewer '
                              f'--run "{run}"')}
    if row is None:
        row = last_training_record(Path(root) / run / "replay_training_stats.jsonl") or {}
    audit = row.get("self_touch_controlled") or row.get("self_touch_audit") or {}
    out["self_touch"] = {
        "patches": row.get("self_contact_patches"),
        "force_share": row.get("self_contact_force_share"),
        "pairs": row.get("self_contact_pairs"),
        "predicted_better_than_shuffled": audit.get("improvement_over_shuffled"),
        "scored_rows": audit.get("rows"),
        "wellness_paid": (row.get("wellness") or {}).get("self_touch"),
        "regions": row.get("skin_regions") or {},
        "regions_view": skin_regions_view(row)}
    out["posture"] = posture_view(row)
    record = read_json_or(live / "record.json", {})
    kept = sorted((live / "rec").glob("frame_*.jpg")) if (live / "rec").is_dir() else []
    out["recording"] = str(record.get("state") or "") == "recording"
    out["kept_frames"] = len(kept)
    out["ready"] = [dict(name=p.name, url=file_url(Path(root), p), seconds=None)
                    for p in sorted((live / "rec").glob("*.mp4"))]
    for entry in read_json_or(live / 'manual_recordings.json', []):
        p = Path(root) / run / 'videos' / Path(entry.get('name', '')).name
        if p.is_file() and p.suffix.lower() == '.mp4':
            out['ready'].append(dict(name=p.name, url=file_url(Path(root), p), seconds=None))
    out["show_sensors"] = bool(read_json_or(live / "viewer_flags.json", {}).get("sensors"))
    if args is None:
        args = read_json_or(Path(root) / run / "args.json", {}) or {}
    return live_texts(out, row, args)


def read_json_or(path, default):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return default


def live_command(root, training, body):
    """The two buttons on the live tabs. Loopback only, like every route here.

    record/stop only write a small file the trainer reads; save turns the frames kept
    while recording into one mp4 under the run's videos folder, under the name the admin
    typed. Nothing here can touch training: the trainer reads the request and keeps
    stepping whatever it says."""
    run = training.get("run")
    if not run:
        raise ValueError("no run is training")
    if training.get("view") == "experiment":
        # the experiment view's live folder is the mirror's copy: its trainer never reads a request written here, and
        # nothing may be written into the experiment's own run folder
        raise ValueError("the live view shows the experiment run through copies; recording, the 3D viewer and the "
                         "sensor display work only for a production run")
    live = Path(root) / run / "live"
    live.mkdir(parents=True, exist_ok=True)
    action = str(body.get("action") or "")
    if action == "sensors":
        want = bool(body.get("show"))
        atomic_write(live / "viewer_flags.json", json.dumps({"sensors": want}))
        return {"show_sensors": want}
    if action == "open_viewer":
        # The window runs on THIS machine, beside the dashboard that serves this page.
        # Detached, with its own console, so closing the page or the server leaves it
        # standing and a crash inside it cannot reach the dashboard.
        import subprocess
        repo = Path(root)
        exe = repo / ".venv_mimo" / "Scripts" / "python.exe"
        if not exe.is_file():
            exe = Path(sys.executable)
        env = dict(os.environ, PYTHONPATH=str(repo / "Source" / "Python"), PYTHONIOENCODING="utf-8")
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        log = open(repo / run / "live" / "viewer.log", "ab", buffering=0)
        proc = subprocess.Popen([str(exe), "-u", "-m", "mimo_bridge.tools.live_viewer",
                                 "--run", str(repo / run)], cwd=str(repo), env=env,
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=flags)
        return {"opened": True, "pid": proc.pid}
    if action in ("record", "stop"):
        state = "recording" if action == "record" else "stopped"
        if action == "record":
            if read_json_or(live / 'record.json', {}).get('state') == 'recording':
                return {'recording': True, 'kept_frames': len(list((live / 'rec').glob('frame_*.jpg')))}
            for old in (live / "rec").glob("frame_*.jpg"):
                old.unlink(missing_ok=True)
        atomic_write(live / "record.json", json.dumps({"state": state, "at": time.time()}))
        if action == "stop":
            return assemble(live)
        return {"recording": True, "kept_frames": 0}
    if action == "save":
        name = re.sub(r"[^A-Za-z0-9 _.-]", "", str(body.get("name") or "")).strip()
        source = live / "rec" / str(body.get("file") or "")
        if not name or not source.is_file() or source.suffix != ".mp4":
            raise ValueError("a recording and a name are needed")
        target = Path(root) / run / "videos" / (name if name.endswith(".mp4") else name + ".mp4")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"{target.name} already exists - choose another name")
        manifest = read_json_or(live / 'manual_recordings.json', [])
        manifest.append(dict(name=target.name, created_at=time.time()))
        atomic_write(live / 'manual_recordings.json', json.dumps(manifest))
        source.replace(target)
        return {"saved": str(target.relative_to(Path(root)).as_posix()), "name": target.name}
    raise ValueError(f"unknown action {action!r}")


def assemble(live):
    """Turn the frames kept while recording into one mp4, then drop the frames."""
    frames = sorted((live / "rec").glob("frame_*.jpg"))
    if not frames:
        return {"recording": False, "kept_frames": 0, "clip": None}
    import imageio.v2 as imageio
    out = live / "rec" / f"unnamed_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    imageio.mimwrite(str(out), [imageio.imread(str(f)) for f in frames], fps=10, quality=6)
    for f in frames:
        f.unlink(missing_ok=True)
    return {"recording": False, "kept_frames": 0,
            "clip": {"file": out.name, "frames": len(frames)}}

# ---------------------------------------------------------------- scheduled physical comparisons
# tools/ability_rechecks.py runs the comparisons of this manifest on saved checkpoints and keeps a
# receipt per attempt in its status file. The Now tab reads the newest receipt of each comparison
# and its result.json here (dashboard staleness check, 2026-09-24: the page showed only their status
# words while its hand-written notes still described production paused at step 380,200).
ABILITY_MANIFEST = 'docs/gain_of_function_v3/ability_rechecks.json'
ABILITY_STATUS = 'Saved/mimo_bridge/ability_rechecks/status.json'
_RESULTS = {}


def _n(value):
    """value when it is a finite number (not a bool), else None."""
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _result_payload(path):
    """A comparison's result.json, read again only when the file changes (the page asks every 30 s)."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _RESULTS:
        if len(_RESULTS) > 64:
            _RESULTS.clear()
        _RESULTS[key] = read_json_or(path, None)
    return _RESULTS[key]


def _deg(value):
    from .plain_numbers import fixed
    return f"{fixed(value, 2)} degrees"


# the outside review's case labels (tools/command_ranking.py CASES) in the page's words. {beat} and
# {not_beat} name the planner (the body card judges two), {reference} the command its search starts from
_RANKING_CASE = {
    'case_1_under_separated': ('its ranking is not measurably better than a coin (the 95% interval reaches 50) and it '
                               'separates the commands less than reality does'),
    'case_2_wrong_order': ('its ranking is not measurably better than a coin (the 95% interval reaches 50) although it '
                           'separates the commands at least as much as reality does'),
    'below_chance': 'it picks the loser more often than a coin would',
    'case_3_planner_not_better': ('it ranks the commands better than a coin, but beyond the sample uncertainty '
                                  '{not_beat} {reference}'),
    'ranking_used': 'it ranks the commands better than a coin, and beyond the sample uncertainty {beat} {reference}',
    'better_than_chance': 'it ranks the commands better than a coin',
}
# the first name of case_1_under_separated (results of commit fd713b068)
_RANKING_CASE['case_1_action_insensitive'] = _RANKING_CASE['case_1_under_separated']


def _per_100_words(block):
    """'68 of 100 pairs of commands', or the pair count itself where 'of 100' would round to 0 or 100."""
    from .plain_numbers import fixed
    value = block['per_100']
    if 1 <= value <= 99:
        return f"{fixed(value, 2)} of 100 pairs of commands"
    right, judged = _n(block.get('pairs_right')), _n(block.get('pairs_judged'))
    if right is None or judged is None:
        return f"{fixed(value, 2)} of 100 pairs of commands"
    return f"{right:g} of {int(judged):,} pairs of commands"


def _interval_words(low, high):
    """The interval's two ends in 100: one decimal when an end lies between 90 and 100 or between 1 and 10
    (a whole number would round it onto 100 or across the chance line), 'under 1' below 1."""
    from .plain_numbers import fixed
    fine = any(90 < v < 100 and v != int(v) for v in (low, high))

    def end(v):
        if 0 < v < 1:
            return 'under 1'
        return f"{v:.1f}" if fine and v != int(v) else fixed(v, 2)
    return f"{end(low)} to {end(high)}"


def _ranking_sentence(block, outcome, reference="the command the planner's search starts from", extra=None,
                      beat="the planner's command beat", not_beat="the planner's command did not beat"):
    """The outside review's statistic in one sentence (a tools/command_ranking.py ranking() block from a
    result file), with its chance floor and its goal line, or None when the block is missing or has no
    value. extra: a last clause (what else the reader needs to weigh the pairs)."""
    from .plain_numbers import pct, times
    if not isinstance(block, dict) or _n(block.get('per_100')) is None:
        return None
    interval = block.get('interval') if isinstance(block.get('interval'), list) else []
    text = f"On {outcome}, the forecast picked the actual winner in {_per_100_words(block)} ("
    if len(interval) == 2 and all(_n(v) is not None for v in interval):
        text += f"95% interval {_interval_words(*interval)} in 100; "
    text += "50 is a coin, 80 to 90 a ranking a planner can optimize)"
    spread = _n(block.get('spread_ratio'))
    if spread is not None:
        text += (f", with the commands' forecasts {pct(spread, 2)} as far apart as their actual outcomes" if spread <= 1
                 else f", with the commands' forecasts {times(spread, 2)} as far apart as their actual outcomes")
    case = 'below_chance' if block.get('below_chance') is True else block.get('case')
    if case in _RANKING_CASE:
        text += ': ' + _RANKING_CASE[case].format(reference=reference, beat=beat, not_beat=not_beat)
    return text + (f"; {extra}" if extra else '') + '.'


# compare_hand_body_goals_2026-09-24.py's witnesses: what each reference command asked the body to do
_BODY_GOALS = {'curl_witness': 'close the hand', 'extension_witness': 'open the hand',
               'wrist_reach_witness': 'raise the left wrist', 'supported_trunk_turn_witness': 'turn the trunk',
               'elbow_flex_witness': 'bend the left elbow'}


def _listed(words):
    return words[0] if len(words) == 1 else ', '.join(words[:-1]) + ' or ' + words[-1]


def _body_words(p):
    """compare_hand_body_goals_2026-09-24.py: the starting controller against the planner searching
    directly and through the learned movement map, on the pose a same-state reference command reached.
    Its error (every body part's position and angle, normalized) is measured 0.25 s after the command
    and compared per body; the paired 95% interval of the mean difference decides "beyond the sample
    uncertainty"."""
    from .plain_numbers import fixed
    s = p['summary']
    vs = s['physical_goal_error']['against_controller']
    n = int(p.get('bodies') or 8)
    hold = _n(p.get('hold_seconds'))
    better = {k: vs[k]['paired_bootstrap95'][1] < 0 for k in ('synergy', 'direct')}
    worse = [f'the {k} planner' for k in ('synergy', 'direct') if vs[k]['paired_bootstrap95'][0] > 0]
    counts = (f"the synergy planner came closer to the goal in {int(vs['synergy']['bodies_better'])} of {n} bodies, "
              f"the direct planner in {int(vs['direct']['bodies_better'])} of {n}")
    if any(better.values()):
        names = ' and '.join(f'the {k} planner' for k in ('synergy', 'direct') if better[k])
        headline = f"{_cap(names)} beat the starting controller beyond the sample uncertainty: {counts}."
        judged = ''
    else:
        headline = f"Neither planner beat the starting controller: {counts}."
        judged = (f"The controller's smaller miss exceeds the sample uncertainty against {' and '.join(worse)}. " if worse
                  else 'No difference from the controller exceeds the sample uncertainty. ')
    hand = s['hand_joint_error_deg']['means']
    goals = [g for g in (p.get('goals') or ['curl_witness', 'extension_witness']) if g in _BODY_GOALS]
    block = p.get('ranking') if isinstance(p.get('ranking'), dict) else {}
    arms = [a for a in (block.get('arms') or []) if isinstance(a, str)]
    optimized = [a for a in (block.get('optimized_arms') or []) if a in arms]
    ranking = _ranking_sentence(block.get('pooled'), "each command's distance from the goal",
                                reference="the starting controller's command (where both searches start)",
                                beat="both planners' commands beat", not_beat="at least one planner's command did not beat",
                                extra=(f"{len(optimized)} of the {len(arms)} commands in each trial were found by "
                                       "searching this same forecast, so its errors favor them" if optimized else None))
    detail = (f"Goal: where a reference command whose target is to {_listed([_BODY_GOALS[g] for g in goals])} actually "
              "took the body from the same saved model step. "
              "The starting controller's command (the motor controller's own proposal, before any planner search) is "
              "compared with the saved model's planner searching toward that pose twice, directly and through the learned "
              "movement map (the synergy planner), keeping the controller's strength and eye commands. The simulator then "
              "measured how far every body part ended from the goal"
              + (f" {fixed(hold, 2)} seconds later" if hold else '') + ". " + judged
              + (ranking + ' ' if ranking else '')
              + _witness_sentence(p, goals, n) +
              f"Hand-joint error: controller {_deg(hand['controller'])}, direct planner {_deg(hand['direct'])}, "
              f"synergy planner {_deg(hand['synergy'])}. Each command ran twice and replayed exactly.")
    return dict(headline=headline, detail=detail, beats=any(better.values()), ranking=ranking,
                suite=str(p.get('goal_suite') or 'hand'),
                limit='No learning happens during the comparison, and no control switch changes because of it.')


def _witness_sentence(p, goals, n):
    """How many bodies' reference commands moved the requested part beyond where the controller's own
    command took it (the result's witness_check, WITNESS_RULE of the comparison), per goal; empty for a
    result written before the check existed."""
    from .plain_numbers import fixed
    counts, mirrored = [], []
    for g in goals:
        row = (p.get('by_goal') or {}).get(g) if isinstance(p.get('by_goal'), dict) else None
        check = (row or {}).get('witness_check') if isinstance(row, dict) else None
        if not isinstance(check, dict) or not isinstance(check.get('informative_bodies'), int) \
                or isinstance(check['informative_bodies'], bool):
            continue
        mirrored.append('mirror_share' in check)
        text = f"{_BODY_GOALS[g]} in {check['informative_bodies']} of {n} bodies"
        got, asked, unit = _n(check.get('achieved_beyond_controller')), _n(check.get('requested_executed')), check.get('unit')
        if got is not None and asked is not None and unit in ('cm', 'degrees'):
            text += (f" (median over the {n} bodies: {fixed(got, 2)} {unit} of the {fixed(asked, 2)} {unit} requested "
                     "once blended with the previous command)")
        counts.append(text)
    if not counts:
        return ''
    # a round written before the mirrored control existed (commit fd713b068) had no reversed request
    than = ("both the controller's own command and the same request reversed did" if all(mirrored)
            else "the controller's own command did")
    return f"The reference command moved the part it asked for further than {than}: {'; '.join(counts)}. "


def _touch_words(p):
    """probe_saved_self_touch_choices.py. The three-layer form (probe 'self_touch_layers_v4', from
    2026-09-25; docs/gain_of_function_v4/agents/self_touch_layers.md) is read by _touch_layer_words. A
    layered result of any other version raises, so the page says only that the result file holds its
    numbers (said_for). Earlier rounds: the contact head's chosen (patch, command) against five
    alternatives on that patch. Counts are read from the result's own counts, never rebuilt from
    fractions of body means."""
    probe = p.get('probe')
    if probe == 'self_touch_layers_v4':
        return _touch_layer_words(p)
    if probe is not None:
        raise ValueError(f'self-touch result version {probe!r} is not read by this page')
    trials = int(p.get('evaluated_held_out_trials') or 0)
    chosen = int(p.get('selected_touches') or 0)
    reachable = int(p.get('selected_patch_physically_reachable_trials') or 0)
    passed = p.get('passed_private_comparison') is True
    if not trials:
        headline = 'No held-out start offered the contact head a choice, so nothing was compared.'
    elif not reachable:
        headline = (f"No command touched the chosen skin patches in any of the {trials} held-out starts, so this round "
                    "cannot tell the choice apart from the alternatives.")
    else:
        headline = (f"The chosen command touched its chosen skin patch in {chosen} of {trials} held-out starts; some command "
                    f"touched that patch in {reachable}. "
                    + ('It beat every alternative beyond the sample uncertainty, the declared pass.' if passed else
                       'It did not beat every alternative beyond the sample uncertainty, the declared pass.'))
    detail = ("Before the simulator runs, the contact head (the network that predicts, for each skin patch, whether "
              "it will be touched in the next 50 ms) picks, among six candidate commands (the controller's, "
              "holding still and four random explorations), the command and untouched skin patch with the highest "
              "predicted chance of new contact; every command is then scored on that patch over the next 50 ms. "
              "Ordinary contact between neighbouring body parts does not count. A held-out start is a saved model step not "
              "used to calibrate the choice. Declared pass: more new contacts than every alternative, beyond the "
              "sample uncertainty over the 8 bodies.")
    return dict(headline=headline, detail=detail, passed=passed, reachable=reachable,
                limit='A one-step choice on a frozen saved model; repeating, keeping and transferring the choice are separate tests.')


def _touch_layer_words(p):
    """The three-layer self-touch round (probe 'self_touch_layers_v4'). The self-touch under test is a
    hand touching another body area. Reachability: could any command make that touch within 50 ms.
    Prediction, judged two ways against a coin flip: 'whether', on the pairs in which the touching
    command asked no body part for a bigger move, did the contact head give it the higher chance; and
    'where', when two commands touched different patches, did the head give each the higher chance on its
    own patch. Selection: when the planner's predicted wellness favoured a touching command over another
    command by more than its tie band, did the planner's objective prefer it to a command of the same
    kind that touched nothing and that the evaluator scored the same. Every number printed is a judged
    score with its goal line of 80 of 100; the unjudged score over every pair is never shown."""
    reach = p['layers']['reachability']
    pred = p['layers']['prediction']['hand_to_body']
    sel = p['layers']['selection']['hand_to_body']
    whether, where = pred['whether'], pred['where']
    starts = int(p.get('evaluated_starts') or 0)
    across = int(reach['reachable_starts_hand_to_body_50ms'])
    within = int(reach['reachable_starts_50ms'])
    whole = int(reach['reachable_starts_hand_to_body_tenure'])
    passed = p.get('passed') is True
    tenure = p.get('tenure_steps')
    order = (f"a whole {50 * tenure} ms command" if isinstance(tenure, int) and not isinstance(tenure, bool) and tenure > 0
             else 'a whole command')

    def of100(stat):
        return f"{round(100 * stat['value'])} of 100"

    def bodies_above(stat, line):
        votes = stat['votes']
        if not votes['bodies_voting']:
            return 'no body had enough pairs to vote'
        return f"{votes['bodies_above']} of {votes['bodies_voting']} bodies with enough pairs were above {line}"

    def verdict(stat, line):
        text = stat['verdict']
        if text in ('above its control', 'not above its control'):
            return text.replace('its control', line) + ' by the declared rule over bodies'
        head, _, why = text.partition(': ')
        return f"{head} ({why})" if why else head
    whether_says = (f"gave the touching command the higher chance of that touch in {of100(whether)} pairs in which it "
                    f"asked no body part for a bigger move, turn or strength than the other command "
                    f"({bodies_above(whether, 'a coin flip')})" if whether['bodies'] else
                    'had no pair in which the touching command asked no body part for a bigger move')
    where_says = (f"when two commands touched different patches, gave each the higher chance on its own patch in "
                  f"{of100(where)} ({bodies_above(where, 'a coin flip')})" if where['bodies'] else
                  'had no two commands that touched different patches and that it told apart')
    if not across:
        headline = (f"No command made a hand touch another body area (the trunk, the head, a leg, or the other hand "
                    f"or arm) within 50 ms in any of the {starts} saved model steps, so this round could not test whether "
                    f"the model foresees or chooses that touch; held for {order}, "
                    + (f"some command did in {whole}." if whole else "none did either.")
                    + (f" Another new touch, such as a finger on its own hand or an arm on the trunk, happened within "
                       f"50 ms in {within} of those model steps." if within else ''))
        lead = ('no command could make a hand touch another body area within 50 ms from the saved model steps, so '
                'self-touch foresight and choice were not tested')
    else:
        if not pred['all_pairs']['bodies']:
            foresight = f"The contact head was not scored: {pred['verdict'].removeprefix('not measured: ')}."
        else:
            both = {'above its control': 'above a coin flip on both, by the declared rule over bodies',
                    'not above its control': 'not above a coin flip on at least one of the two, by the declared rule '
                                             'over bodies'}.get(pred['verdict'])
            foresight = (f"The contact head {whether_says}, and {where_says}; 50 of 100 is a coin flip and the goal "
                         f"line, a ranking a planner can use, is 80. Foresight verdict: "
                         f"{both or verdict(pred, 'a coin flip')}.")
        if not sel['bodies']:
            choice = f"The planner's choice was not scored: {sel['verdict'].removeprefix('not measured: ')}."
        else:
            moves = sel['move_relation']
            choice = (f"When the planner's predicted wellness favoured a touching command over another command by more "
                      f"than its tie band, the planner's objective preferred it to a command of the same kind from "
                      f"the same model step that touched nothing and that the evaluator scored the same, in {of100(sel)} "
                      f"such pairs (50 = a coin flip, goal line 80; the touching command asked the smaller move in "
                      f"{moves['smaller']} of the {sel['pairs']} pairs and the bigger one in {moves['bigger']}); "
                      f"{bodies_above(sel, 'a coin flip')}: {verdict(sel, 'a coin flip')}.")
        headline = (f"Some command made a hand touch another body area within 50 ms in {across} of {starts} saved "
                    f"model steps. {foresight} {choice}")
        if passed:
            lead = (f"the contact head gave the touching command the higher chance ({of100(whether)} pairs in which it "
                    f"asked no bigger move) and gave each of two touching commands the higher chance on its own patch "
                    f"({of100(where)}), and the planner's objective preferred the touching command to a same-kind "
                    f"command that did not touch in {of100(sel)} pairs (goal line 80 for each)")
        elif not pred['all_pairs']['bodies']:
            lead = 'self-touch foresight and choice were not measured: ' + pred['verdict'].removeprefix('not measured: ')
        else:
            head = (f"gave the touching command the higher chance in {of100(whether)} pairs in which it asked no bigger "
                    "move" if whether['bodies'] else 'had no pair in which the touching command asked no bigger move')
            patch = (f"gave each of two touching commands the higher chance on its own patch in {of100(where)}"
                     if where['bodies'] else 'had no two touching commands to compare')
            planner = (f"the planner's objective preferred the touching command to a same-kind command that did not "
                       f"touch in {of100(sel)} pairs" if sel['bodies'] else "the planner's choice was not measured")
            lead = (f"self-touch foresight and choice did not both pass: the contact head {head} and {patch} (50 = a "
                    f"coin flip, goal line 80); {planner}")
    detail = ("At each saved model step the same state is replayed with the controller's command, holding still, four "
              "explorations, the command production itself sends, a finger curl per hand, and commands that move each "
              "hand's target toward its two nearest untouched patches of another body area. A new touch is a contact "
              "between the body's own parts that began within 50 ms, the horizon of the contact head (the network "
              "that predicts, for each skin patch, whether it will be touched in the next 50 ms); ordinary "
              "contact between neighbouring parts does not count. The self-touch under test is a hand touching "
              "another body area: the trunk, the head, a leg, or the other hand or arm. The contact head is judged "
              "two ways. First, on pairs of a command that made that touch on a patch and one that left the patch "
              "alone, in which the touching command asked no body part for a bigger move, turn or strength: a head "
              "whose chances only grow with how much a body part is asked to move scores at most 50 there. Second, "
              "on two commands that touched different patches, half a point for each given the higher chance on its "
              "own patch: a head that ranks every patch by one measure of the command scores exactly 50 there. The "
              "planner is judged by comparing each touching command with a command of the same kind from the same "
              "model step that touched nothing and that the evaluator scored the same within its tie band; this shows "
              "the planner preferring the commands that touched, and the page says how often the touching command "
              "was also the smaller move, since a planner that prefers smaller moves would show the same. Declared "
              "pass: all three above 50 by an exact sign rule over the bodies with at least 5 pairs from 2 touches: "
              "every one of 5, 6 or 7 such bodies, or 7 of 8.")
    return dict(headline=headline, detail=detail, passed=passed, reachable=across, lead=lead, form='layers',
                limit='A one-step comparison on a frozen saved model; repeating, keeping and transferring the '
                      'choice are separate tests.')


def _gaze_words(p):
    """probe_saved_gaze.py: the eye-angle forecast 150 ms ahead against predicting no eye movement, and
    the predicted view with the actual eye command against the same prediction with the eyes held.
    'saved_starts' is one saved moment per body; 'after_head_command' is a second start per body taken
    after a head turn."""
    from .plain_numbers import pct
    s = p['summary']['saved_starts']['150ms']
    eye, view = s['eye_prediction_error_degrees'], s['visual_prediction_mse']
    turned = ((p['summary'].get('after_head_command') or {}).get('150ms') or {}).get('eye_prediction_error_degrees') or {}

    def less(fraction):
        return f"{pct(fraction, 2)} less error" if fraction >= 0 else f"{pct(-fraction, 2)} more error"
    headline = (f"Eye movement 150 ms ahead was predicted with {less(eye['error_reduction_fraction'])} than assuming the "
                f"eyes stay still ({_deg(eye['candidate_mean'])} against {_deg(eye['comparison_mean'])}).")
    beyond = [name for name, row in (('eye-angle', eye), ('view', view))
              if row.get('candidate_error_lower_beyond_sample_uncertainty') is True]
    judged = ('Both differences exceed the sample uncertainty.' if len(beyond) == 2 else
              f"Only the {beyond[0]} difference exceeds the sample uncertainty." if beyond else
              'Neither difference exceeds the sample uncertainty.')
    detail = ("The auditor (the test script, not the model) sends left, right, up and down eye commands from one saved "
              "model step in each of the 8 bodies. "
              f"Knowing the actual eye command gave the predicted view {less(view['error_reduction_fraction'])} than the same "
              f"prediction told the eyes were held still. {judged}"
              + (f" From a second start after a head turn, the eye forecast had {less(turned['error_reduction_fraction'])}."
                 if _n(turned.get('error_reduction_fraction')) is not None else ''))
    return dict(headline=headline, detail=detail, beats=eye.get('candidate_error_lower_beyond_sample_uncertainty') is True,
                tile=less(eye['error_reduction_fraction']),
                limit='The auditor chooses the commands, so this measures prediction, not useful looking.')


_HAND_GOALS = {'curl': 'closing', 'extension': 'opening', 'index_selective': 'index finger alone'}


def _hand_words(p):
    """qualify_hand_execution_2026-09-24.py: mean hand-joint error after 1 second with the
    wrist-following hand targets against the old fixed-world targets; no learner is loaded. Only the
    wrist-following run is repeated for the exact-replay check."""
    from .plain_numbers import fixed
    s = p['summary']
    goals = [g for g in _HAND_GOALS if g in s]
    new, old = [s[g]['moving_wrist'] for g in goals], [s[g]['fixed_world'] for g in goals]
    headline = (f"Hand targets that follow the wrist ended {fixed(min(new), 2)} to {fixed(max(new), 2)} degrees from the "
                f"finger goals after 1 second, against {fixed(min(old), 2)} to {fixed(max(old), 2)} degrees with the old "
                "fixed-world targets.")
    detail = ("Mean hand-joint error after 1 second across the 8 saved bodies, wrist-following against fixed-world: "
              + '; '.join(f"{_HAND_GOALS[g]} {_deg(s[g]['moving_wrist'])} against {_deg(s[g]['fixed_world'])}" for g in goals)
              + f". {int(p['comparisons'])} comparisons; the wrist-following run of each replayed exactly.")
    return dict(headline=headline, detail=detail, beats=all(a < b for a, b in zip(new, old)),
                limit='Researcher-set goals and no learner: this measures what the hand controls can physically do, not a learned choice.')


def _scale_pct(value):
    """A difference on the 0-to-5 wellness sum (five terms, each 0 to 1) as a percentage of that scale."""
    from .plain_numbers import pct
    return pct(float(value) / 5.0, 2)


def _scale_share(value):
    return f"{_scale_pct(value)} of the 0-to-5 wellness scale"


def _interval_pct(stat):
    from .plain_numbers import pct
    lo, hi = stat['interval']
    return f"{pct(lo, 2)} to {pct(hi, 2)}"


def _script_label(p):
    """The first 8 characters of the SHA-256 of the probe script that measured a round (the result's
    script_sha256, the scheduler's digest of the same file), or None when the result does not record it."""
    sha = p.get('script_sha256') if isinstance(p, dict) else None
    return sha[:8] if isinstance(sha, str) and len(sha) >= 8 else None


def _checkpoint_step(p):
    match = re.search(r'ckpt_step_(\d+)', str((p or {}).get('checkpoint') or ''))
    return int(match.group(1)) if match else None


def _script_sentence(read):
    """Which versions of a probe script measured the rounds: each labelled by the first 8 characters of the
    script's SHA-256, with its round count and checkpoint steps (the own-command analysis review of 2026-10-02:
    three versions of the planner-candidate probe produced its nine rounds, and a reading can change meaning
    between versions)."""
    groups = {}
    for w in read:
        groups.setdefault(w.get('script') or 'unrecorded', []).append(w.get('checkpoint_step'))
    def where(steps):
        steps = sorted(s for s in steps if isinstance(s, int))
        if not steps:
            return ''
        return f" (step {steps[0]:,})" if len(steps) == 1 else f" (steps {steps[0]:,} to {steps[-1]:,})"
    if len(groups) == 1:
        (label, steps), = groups.items()
        if label == 'unrecorded':
            return " The result files do not record which version of the probe script measured them."
        return (f" {'The round comes' if len(steps) == 1 else f'All {len(steps)} rounds come'} from one version of the "
                f"probe script, {label} (the first 8 characters of its SHA-256).")
    parts = [f"{label}, {len(steps)} round{'s' if len(steps) != 1 else ''}{where(steps)}" for label, steps in groups.items()]
    return (f" The rounds come from {len(groups)} versions of the probe script, labelled by the first 8 characters of "
            f"its SHA-256 ('unrecorded' = a result file without it): {'; '.join(parts)}. A reading can change meaning "
            "between versions.")


def _planner_words(p):
    """docs/gain_of_function_v4/probe_planner_candidates.py: where along the planner's route the
    difference between five candidate commands is lost (its declared rules, DECISION_RULES):
    the forward model's forecast, the approximate evaluator (Judge B) scoring the ACTUAL outcomes,
    or the planner's selection; and whether looking 150 ms ahead instead of 50 ms picks a command
    with better actual wellness. Production's 'tied' readings are not this probe's verdict: a tie is a
    predicted difference inside the evaluator's own error band, which was wider than the spread it predicts
    between candidates in 1,174 of 1,174 report windows of H84 since Sep 23, median 12.6 times the spread
    (docs/gain_of_function_v4/agents/hold_loss_history_2026-10-02.json), so the text gives the round's band
    over its spread and points to the forecast's captured share as the planner-side reading of the forward
    model (own-command analysis review, 2026-10-02)."""
    from .plain_numbers import fixed, pct
    v, s = p['verdict'], p['summary']
    fm, ev = s['forward_model']['transfer_judge_units'], s['evaluator']['transfer']
    rank = s['planner']['chosen_rank_by_predicted_wellness']
    chance = _n(s['planner'].get('chance_rank')) or 3.0
    k = len(p.get('candidates') or ()) or 5
    first = str(v['first_flat_stage_along_the_route'])
    fm_said = (f"the forecast carried {pct(fm['value'], 2)} of the difference the candidates' actual outcomes "
               "made (0% = it ignores the command, 100% = it carries the whole difference)")
    ev_said = (f"the evaluator, scoring the actual outcomes, saw {pct(ev['value'], 2)} of the true wellness "
               "difference (0% = blind to it, 100% = sees all of it)")
    if first == 'forward model':
        headline = f"The planner recheck's first flat stage is the forward model: {fm_said}."
        lead = "the planner recheck's first flat stage is the forward model's forecast"
    elif first == 'evaluator':
        headline = f"The planner recheck's first flat stage is the evaluator: {ev_said}."
        lead = "the planner recheck's first flat stage is the evaluator"
    elif first == 'planner':
        headline = (f"Forecast and evaluator pass differences on, but the planner does not choose by predicted wellness: "
                    f"its chosen command ranks {fixed(rank['value'], 2)} of {k} by predicted wellness ({fixed(chance, 2)} = "
                    "chance, 1 = always the best).")
        lead = "the planner recheck's first flat stage is the planner's own selection"
    else:
        headline = f"No stage of the planner's route is flat: {fm_said}; {ev_said}."
        lead = "no stage of the planner's route is flat"
    # the planner-side reading of the forward model: the captured share pooled over Judge B's senses, 50 ms ahead
    pooled = (((s['forward_model'].get('senses') or {}).get('planner') or {}).get('1') or {}).get('judge_senses') or {}
    share = pooled.get('captured_share') if isinstance(pooled.get('captured_share'), dict) else {}
    if share.get('measurable') is False:
        captured = ("The real difference between the candidates' next tokens was within the tokens' float32 floor, so "
                    "the forecast's captured share is not measurable this round.")
    elif _n(share.get('value')) is not None:
        captured = (f"The forecast captured {pct(share['value'], 2)} of the real difference between two candidates' "
                    "next tokens (pooled over vision, tactile, audio and proprioception, 50 ms ahead; 0% = none of it, "
                    "100% = all of it"
                    + (f"; 95% interval {_interval_pct(share)}" if share.get('interval') else '') + ").")
    else:
        captured = None
    stage = first if first in ('forward model', 'evaluator', 'planner') else 'no flat stage'
    band, dec = s['tie_band'], s['tie_band'].get('decomposition') or {}
    depth = s['depth']['wellness']
    gain = depth['wellness_tenure_depth3_minus_depth1']
    # the outside review's statistic (planner_candidates_v2): the forecast of distance to the planner's goal
    # against the actual distance when the order's tenure ends
    end = (s.get('ranking') if isinstance(s.get('ranking'), dict) else {}).get('goal_distance_end')
    rest = end.get('without_chosen') if isinstance(end, dict) and isinstance(end.get('without_chosen'), dict) else None
    ranking = _ranking_sentence(end, "distance to the planner's goal (the remembered model step it steers toward) when the "
                                "command ends", reference="the controller's own proposal (where its search starts)",
                                extra=(None if rest is None or _n(rest.get('per_100')) is None else
                                       "the planner's own command was found by searching this same forecast, and over the "
                                       f"other four it picked the actual winner in {_per_100_words(rest)}"))
    parts = [f"{int(p['evaluated_starts'])} saved planning events ({int(p['bodies'])} bodies), {k} candidate commands "
             "each: the planner's own, holding still, the controller's proposal and two small perturbations, all "
             "executed from the same saved state for the whole hold of the command."] + ([ranking] if ranking else []) + [
             f"Forecast transfer {pct(fm['value'], 2)} (95% interval {_interval_pct(fm)}); evaluator transfer "
             f"{pct(ev['value'], 2)} (95% interval {_interval_pct(ev)}).",
             f"The tie band (the miss between a predicted and the true wellness difference that 90 of every 100 "
             f"candidate pairs stay within) is {_scale_share(band['remeasured_planner_route'])} here; production uses "
             f"{_scale_share(band['production'])}."]
    if _n(dec.get('forward_model_share')) is not None and _n(dec.get('evaluator_share')) is not None:
        parts.append(f"Of that miss, the forecast accounts for {pct(dec['forward_model_share'], 2)} and the evaluator for "
                     f"{pct(dec['evaluator_share'], 2)}, the rest their interaction.")
    # WHAT A PRODUCTION TIE IS (own-command analysis review, 2026-10-02): the chosen command's predicted wellness sum
    # within the production band of the best alternative's. The band is the evaluator's own error on a difference;
    # while it is wider than the spread of the candidates' predicted sums every choice reads tied, whatever the
    # forward model predicts, so the tie count is not evidence about the forward model
    spread = _n(((s.get('evaluator') or {}).get('candidate_spread_median') or {}).get('predicted'))
    prod_band = _n(band.get('production'))
    tie = None
    if spread is not None and prod_band is not None:
        if spread > 0 and prod_band >= spread:
            tie = (f"A production tie is a predicted difference inside the production band, the evaluator's own "
                   f"error on a difference between two candidates; that band is {fixed(prod_band / spread, 2)} "
                   "times the spread the evaluator predicted between the five candidates here (median over the "
                   "starts), so every production choice reads tied whatever the forward model predicts: the tie "
                   "count is no evidence about the forward model.")
        elif spread > 0:
            tie = ("The spread the evaluator predicted between the five candidates exceeded the production band "
                   "here, so a production choice could leave the tie band.")
        else:
            tie = ("The evaluator predicted the same wellness for all five candidates here, so every production "
                   "choice reads tied whatever the forward model predicts: the tie count is no evidence about the "
                   "forward model.")
        parts.append(tie)
    if captured:
        captured += "" if share.get('measurable') is False else " This captured share is the planner-side reading of the forward model."
        parts.append(captured)
    parts.append(f"Looking 150 ms ahead instead of 50 ms changed the pick at {pct(depth['differs_share'], 2)} of the "
                 f"planning events and changed actual wellness over the command's hold by {_scale_share(gain['mean'])} "
                 f"(95% interval {_scale_pct(gain['interval'][0])} to {_scale_pct(gain['interval'][1])}); "
                 + ("a better physical decision, the declared pass." if p.get('passed') is True else
                    "not a better physical decision by the declared rule (interval above zero)."))
    script = _script_label(p)
    detail = ' '.join(parts) + (f" Measured by the probe script {script} (the first 8 characters of its SHA-256"
                                + (f", {p['probe']}" if isinstance(p.get('probe'), str) else '') + ")." if script else '')
    # tie and captured: the two sentences of the detail a goal-catalogue card binds on their own (shows 'tie' and
    # 'captured'), so a card that prints the production tie count also prints what a tie is and the planner-side reading
    return dict(headline=headline, detail=detail, passed=p.get('passed') is True, first=stage, lead=lead,
                tie=tie, captured=captured,
                ranking=ranking, script=script, probe=p.get('probe'), checkpoint_step=_checkpoint_step(p),
                limit=f"A frozen saved model at {int(p['evaluated_starts'])} saved planning events; the probe changes no "
                      "weight and no control switch.")


def _hand_memory_words(p):
    """docs/gain_of_function_v3/compare_hand_memory_route_2026-09-24.py: the learner's remembered
    hand-goal stage on against off at two ordinary motor-latch decisions of each held-out body 4-7.
    Hand-shape error: each fingertip segment's position and rotation error from the remembered
    shape, squared after dividing by 1.80 cm and 11 degrees (lem.model.efference EFFERENCE_SCALES)."""
    from .plain_numbers import fixed, pct
    s = p['summary']
    shape, where, turn = s['hand_shape_error'], s['finger_position_error_cm'], s['finger_orientation_error_deg']
    base, hm = shape['means']['baseline'], shape['means']['hand_memory']
    change = 1.0 - hm / base
    bodies = len(p.get('held_out_bodies') or ()) or 4
    closer = (f"{pct(change, 2)} closer to" if change >= 0 else f"{pct(-change, 2)} farther from")
    headline = (f"With the learned hand stage the hand ended {closer} the remembered hand shape than the same decision "
                f"without it; {int(shape['bodies_improved'])} of {bodies} held-out bodies improved, and the round "
                + ("passed its declared admission." if p.get('passed') is True else "did not pass its declared admission."))
    detail = (f"Two ordinary movement decisions in each held-out body (bodies not used to fit the hand calibration): the "
              "goal is the shape the planner itself picked from memory, pursued with the hand stage on and off; every "
              "command except the fingers' is identical, and each branch replayed exactly. Fingertip position error "
              f"{fixed(where['means']['hand_memory'], 2)} cm against {fixed(where['means']['baseline'], 2)} cm; fingertip "
              f"angle error {_deg(turn['means']['hand_memory'])} against {_deg(turn['means']['baseline'])}. Declared "
              f"admission: all {bodies} bodies improve and the paired 95% interval stays below zero.")
    return dict(headline=headline, detail=detail, passed=p.get('passed') is True,
                tile=(f"{pct(change, 2)} less hand-shape error" if change >= 0 else f"{pct(-change, 2)} more hand-shape error"),
                lead=('the learned hand stage brought hands closer to remembered shapes' if p.get('passed') is True else
                      'the learned hand stage did not pass its admission'),
                limit='A frozen saved model; repeating the choice after further learning and in changed conditions are separate tests.')


_RESULT_WORDS = {'body_and_synergy': _body_words, 'chosen_self_touch': _touch_words,
                 'gaze_prediction': _gaze_words, 'hand_following': _hand_words,
                 'planner_candidates': _planner_words, 'hand_memory_route': _hand_memory_words}
# the newest round in goal terms, for the panel's lead sentence (a result may name its own 'lead')
_LEAD_WORDS = {
    'body_and_synergy': ('the learned planners beat the starting controller', 'the learned planners did not beat the starting controller'),
    'chosen_self_touch': ('the chosen self-touch beat its alternatives', 'the chosen self-touch did not beat its alternatives'),
    'gaze_prediction': ("eye forecasts beat 'the eyes stay still'", "eye forecasts did not beat 'the eyes stay still'"),
    'hand_following': ('wrist-following hand targets followed better than the old targets (a controls test)',
                       'wrist-following hand targets did not follow better than the old targets'),
}


def _won(task_id, words):
    """Whether a round met its comparison's own line: a declared pass, or beating its reference."""
    return bool(words.get('passed') if task_id in ('chosen_self_touch', 'planner_candidates', 'hand_memory_route')
                else words.get('beats'))


def _trend(task_id, read, since):
    """The rounds so far in one sentence, in each comparison's own words."""
    k, n = sum(_won(task_id, w) for w in read), len(read)
    if task_id == 'body_and_synergy':
        text = (f"In {n} completed rounds{since}, neither planner beat the controller beyond the sample uncertainty." if not k
                else f"A planner beat the controller beyond the sample uncertainty in {k} of {n} completed rounds{since}.")
        body = sum(1 for w in read if w.get('suite') == 'hand_and_body')
        if body and body < n:
            text += (f" {body} of those rounds also had the wrist, trunk and elbow goals; "
                     + ("the other" if n - body == 1 else f"the other {n - body}") + " the hand goals alone.")
        return text
    if task_id == 'chosen_self_touch':
        # an earlier-form round compared one chosen patch; a layered round is a different declared test that
        # asks for a hand touching another body area, so each form is counted in its own words
        earlier = [w for w in read if w.get('form') != 'layers']
        layered = [w for w in read if w.get('form') == 'layers']
        text = ''
        if earlier:
            empty = sum(1 for w in earlier if not w.get('reachable'))
            won = sum(_won(task_id, w) for w in earlier)
            text = (f"Passed its declared comparison in {won} of {len(earlier)} completed rounds{since}."
                    if not layered else
                    f"The earlier single-choice comparison passed in {won} of {len(earlier)} completed rounds{since}.")
            text += f" In {empty} of them no command touched the skin being compared." if empty else ''
        if layered:
            empty = sum(1 for w in layered if not w.get('reachable'))
            won = sum(_won(task_id, w) for w in layered)
            text += ((' ' if text else '') + f"The three-layer self-touch test passed in {won} of {len(layered)} "
                     f"completed rounds{'' if earlier else since}."
                     + (f" In {empty} of them no command made a hand touch another body area within 50 ms."
                        if empty else ''))
        return text
    if task_id == 'gaze_prediction':
        return f"Eye-angle prediction beat 'the eyes stay still' beyond the sample uncertainty in {k} of {n} completed rounds{since}."
    if task_id == 'hand_following':
        return f"Wrist-following targets had less error on all three goals in {k} of {n} completed rounds{since}."
    if task_id == 'planner_candidates':
        stages = {}
        for w in read:
            stages[w.get('first')] = stages.get(w.get('first'), 0) + 1
        named = '; '.join(f"{name}: {count} round{'s' if count != 1 else ''}" for name, count in stages.items() if name)
        # each round labelled by the probe script that measured it (the series spans versions of the probe)
        return (f"First flat stage in {n} completed round{'s' if n != 1 else ''}{since}: {named}. Looking further ahead "
                f"gave a better physical decision in {k} of {n}." + _script_sentence(read))
    if task_id == 'hand_memory_route':
        return f"Passed its declared admission in {k} of {n} completed rounds{since}."
    return None


_HELD_STEP = re.compile(r'\s*\((?:first_step|first step)\b[^)]*\)')
_ACTIVE_TASK = ('starting', 'running', 'process identity unavailable')


def held_note(task):
    """The reason a manifest task is held from scheduling, or None. A hold sets the task's first step
    out of reach (999,999,999 on 2026-09-24) and writes the reason under 'held'; the page shows the
    reason and never the out-of-reach step ("First round at step 999,999,999")."""
    note = str((task or {}).get('held') or '').strip() if isinstance(task, dict) else ''
    if not note:
        return None
    note = _HELD_STEP.sub('', note)
    return re.sub(r'\bfirst_step (\d+)', lambda m: f"its schedule from step {int(m.group(1)):,}", note)


def with_holds(root, checks):
    """The scheduler's status with each held manifest task shown as held: no due step and the
    manifest's reason. The scheduler keeps the last status it wrote for a held task (the eye test
    still read "waiting for requested checkpoint" after its hold at 9:14 PM PDT on 2026-09-24); a
    task it is still running keeps its own status."""
    manifest = read_json_or(root/ABILITY_MANIFEST, {})
    holds = {t['id']: held_note(t) for t in manifest.get('tasks', []) if isinstance(t, dict) and t.get('id')}
    holds = {k: v for k, v in holds.items() if v}
    if not holds or not isinstance(checks, dict):
        return checks
    out, tasks = dict(checks), dict(checks.get('tasks') or {})
    for task_id, note in holds.items():
        record = dict(tasks.get(task_id) or {})
        record.update(due_step=None, held=note)
        if record.get('status') not in _ACTIVE_TASK:
            record.update(status='held', reason=note)
        tasks[task_id] = record
    out['tasks'] = tasks
    return out


def _overdue_round(root, due):
    """The next-round sentence of a round whose due step training has already passed, else None: the due step, the
    newest training record's step and whether production training is paused (the supervisor's pause command or pause
    reason, or a status other than 'training'). the dashboard decision of Oct 21, 2026: an overdue round's text
    says what is due and why it has not run, never a step training has already passed as if it were ahead."""
    if due is None:
        return None
    data, _run = _active_training_record(root)
    step = _n((data or {}).get('step'))
    if step is None or due > step:
        return None
    supervisor = read_json_or(root/'Saved/mimo_bridge/training_continuation.json', {})
    control = read_json_or(root/'Saved/mimo_bridge/supervisor_control.json', {})
    paused = (str(control.get('cmd')) == 'pause' or bool(supervisor.get('pause_reason'))
              or str(supervisor.get('status')) != 'training')
    return (f"Due at step {int(due):,}; production training stands at step {int(step):,} and is "
            + ("paused; this round has not run." if paused else "running; this round has not run yet."))


def ability_results(root, checks=None):
    """The newest completed result of every scheduled comparison, in plain sentences, with the
    result file, its checkpoint step and its AM/PM completion time, the count of rounds, the next
    round and one lead sentence in goal terms."""
    checks = with_holds(root, checks if checks is not None else read_json_or(root/ABILITY_STATUS, {}))
    manifest = read_json_or(root/ABILITY_MANIFEST, {})
    declared = [t for t in manifest.get('tasks', []) if isinstance(t, dict) and t.get('id')]
    tasks = {k: v for k, v in ((checks or {}).get('tasks') or {}).items() if isinstance(v, dict)}
    rows, total, first = [], 0, None
    for task in declared or [dict(id=k) for k in tasks]:
        record = tasks.get(task['id']) or {}
        history = [h for h in (record.get('history') or []) if isinstance(h, dict)]
        finished = []
        for h in history:
            if h.get('status') != 'complete' or not h.get('result'):
                continue
            try:
                path = confined(root, h['result'])
            except (TypeError, ValueError, OSError):
                continue
            payload = _result_payload(path)
            if isinstance(payload, dict):
                finished.append((h, path, payload))
        finished.sort(key=lambda f: (_n(f[0].get('attempt_step')) or 0, _n(f[0].get('completed_at')) or 0))
        # a probe's refusal of an out-of-date scheduler, taken back by a current one, is no failed attempt
        failed = sum(1 for h in history if h.get('status') == 'failed' and not h.get('stale_refusal_forgiven'))
        steps = [int(f[0]['attempt_step']) for f in finished if _n(f[0].get('attempt_step')) is not None]
        held = held_note(task)
        due = None if held else (_n(record.get('due_step')) or _n(task.get('first_step')))
        # 'frequent' repeats every every_steps; 'triggered' runs on a declared trigger and at least every
        # every_steps (ability_rechecks, the operator's decision of 2026-09-25)
        row = dict(id=task['id'], title=record.get('title') or task.get('title') or task['id'], status=record.get('status'),
                   due_step=due, every_steps=_n(task.get('every_steps')), completed=len(finished), failed=failed,
                   cadence=('triggered' if task.get('class') == 'triggered' else 'frequent'),
                   # the scheduler's warning (ability_rechecks task_warning): an unmeasured mechanism that steers
                   # production, or a test running far longer than any completed round
                   warning=record.get('warning') if isinstance(record.get('warning'), str) else None,
                   first_step=steps[0] if steps else None, held=held,
                   next=((f"A round is running now; no later round is scheduled. {held}"
                          if record.get('status') in _ACTIVE_TASK else f"No round is scheduled. {held}") if held else
                         (_overdue_round(root, due) or (f"Next round at step {int(due):,}." if finished else
                                                        f"First round at step {int(due):,}.")) if due else None))
        words = _RESULT_WORDS.get(task['id'])

        def said_for(payload):
            try:
                return words(payload) if words else None
            except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
                return None
        if finished:
            h, path, payload = finished[-1]
            said = said_for(payload)
            started, ended = _n(h.get('started_at')), _n(h.get('completed_at'))
            row['latest'] = dict(step=h.get('attempt_step'), completed_at=ended, when=clock(ended),
                                 seconds=(ended-started) if started and ended and ended > started else None,
                                 url=file_url(root, path), described=said is not None,
                                 **(said or dict(headline='Completed; the result file holds its numbers.', detail='', limit='')))
            read = [w for w in (said_for(f[2]) for f in finished) if w is not None]
            text = _trend(task['id'], read, f" since step {steps[0]:,}" if steps else '') if read else None
            if text is not None:
                if len(read) < len(finished):
                    text += f" {len(finished)-len(read)} result file{'s' if len(finished)-len(read) != 1 else ''} could not be read."
                if failed:
                    text += f" {failed} further attempt{'s' if failed != 1 else ''} ended without a result; the scheduler keeps each log."
                row['trend'] = text
            if task['id'] == 'body_and_synergy' and read:
                row['tile'] = f"{sum(_won(task['id'], w) for w in read)} of {len(read)} rounds won"
            elif row['latest'].get('tile'):
                row['tile'] = row['latest']['tile']
        total += len(finished)
        first = min([s for s in (first, row['first_step']) if s is not None], default=None)
        rows.append(row)
    summary = (f"{total} scheduled physical comparisons have finished on saved checkpoints of this lineage since step "
               f"{first:,}; the newest result of each is on the Now tab. A finished comparison records an outcome; it "
               "does not pass an ability or change a control switch." if total else None)
    good, bad, empty = [], [], []
    for r in rows:
        latest = r.get('latest') or {}
        if not latest.get('described'):
            continue
        if r['id'] == 'chosen_self_touch' and not latest.get('reachable'):
            # the layered round's own lead says what could not be reached; the earlier form has none
            empty.append(latest.get('lead') or 'no command touched the skin being compared in the self-touch round')
            continue
        won = _won(r['id'], latest)
        phrase = latest.get('lead') or (_LEAD_WORDS.get(r['id']) or (None, None))[0 if won else 1]
        if phrase:
            (good if won else bad).append(phrase)
    lead = ('In the newest rounds, ' + '; '.join(good + bad + empty) + '.') if good or bad or empty else None
    return dict(rows=rows, completed=total, comparisons=len([r for r in rows if r['completed']]), first_step=first,
                summary=summary, lead=_cap(lead) if lead else None,
                source=file_url(root, root/ABILITY_STATUS) if (root/ABILITY_STATUS).is_file() else None)


def _launch_record(stats):
    """A run's first launch record (type 'launch', with the resumed step and launch time)."""
    try:
        with open(stats, 'rb') as stream:
            for line in stream.read(262_144).split(b'\n'):
                if b'"type": "launch"' in line[:48]:
                    row = json.loads(line)
                    return row if isinstance(row, dict) else {}
    except (OSError, ValueError):
        pass
    return {}


def _segment(name):
    match = re.search(r'_seg(\d+)', str(name or ''))
    return f"segment {int(match.group(1))}" if match else str(name or 'the active run')


def production_activity(root, checks=None, now=None):
    """The Now card's present work, read from the supervisor record, the active run's newest record
    and the comparison scheduler - never from a run path pinned in the hand-written catalogue."""
    from .plain_numbers import fixed
    from .training_brief import duration
    now = time.time() if now is None else now
    saved = root/'Saved/mimo_bridge'
    supervisor = read_json_or(saved/'training_continuation.json', {})
    try:
        run = confined(root, supervisor['active_run'])
    except (KeyError, TypeError, ValueError, OSError):
        return None
    record = last_training_record(run/'replay_training_stats.jsonl')
    # right after a relaunch the run holds only its launch record, whose step is the resumed
    # checkpoint's: the card says so instead of falling back to the catalogue's old state
    launched = not record
    if launched:
        record = _launch_record(run/'replay_training_stats.jsonl')
    step, stamp, rate = _n(record.get('step')), _n(record.get('timestamp')), _n(record.get('model_steps_per_s'))
    if step is None:
        return None
    step = int(step)
    # a held task has no due step, so it is never the next or the later comparison
    checks = with_holds(root, checks if checks is not None else read_json_or(saved/'ability_rechecks/status.json', {}))
    manifest = read_json_or(root/ABILITY_MANIFEST, {})
    declared = {t.get('id') for t in manifest.get('tasks', []) if isinstance(t, dict) and t.get('id')}
    every = {t.get('id'): _n(t.get('every_steps')) for t in manifest.get('tasks', []) if isinstance(t, dict)}
    # the manifest decides which comparisons exist; a task it no longer lists is not scheduled
    tasks = {k: v for k, v in ((checks or {}).get('tasks') or {}).items()
             if isinstance(v, dict) and (not declared or k in declared)}
    window = (checks or {}).get('comparison_window') or {}
    checks_stamp = _n((checks or {}).get('timestamp'))
    checkpoint = re.fullmatch(r'ckpt_step_(\d+)(?:_.*)?', Path(str(window.get('checkpoint') or '')).name)
    checkpoint = int(checkpoint.group(1)) if checkpoint else None
    open_window = (window.get('state') in ('preparing', 'saving', 'ready') and checks_stamp is not None
                   and now-checks_stamp < 180)
    # comparisons of the open window not yet finished on its checkpoint; finished ones already carry
    # their next due step
    in_window = {k for k in (window.get('targets') or {}) if checkpoint is None
                 or (_n((tasks.get(k) or {}).get('completed_step')) or -1) < checkpoint} if open_window else set()
    window_tasks = set(window.get('targets') or {}) if open_window else set()
    supervisor_stamp = _n(supervisor.get('timestamp'))
    running = (str(supervisor.get('status')) == 'training' and stamp is not None and 0 <= now-stamp < 600
               and supervisor_stamp is not None and now-supervisor_stamp < 180)
    reference = checkpoint if open_window and checkpoint else step
    pending = sorted({int(t['due_step']) for k, t in tasks.items() if k not in in_window and _n(t.get('due_step')) is not None})

    def group(due):
        return [t for k, t in tasks.items() if k not in in_window and _n(t.get('due_step')) == due]
    spans = [s for s in every.values() if s]
    uniform = spans[0] if spans and all(s == spans[0] for s in spans) else None
    segment = _segment(run.name)
    activity = dict(kind='production', title=f"Production training, {segment}", run=run.relative_to(root).as_posix(),
                    run_name=run.name, step=step, recorded_at=stamp, rate=rate, done=None, total=None,
                    seconds_remaining=None, unit='training updates toward the next scheduled comparison',
                    purpose=(f"Production training runs in {segment} ({run.name}). The scheduler tests frozen saved "
                             "copies of the model " + (f"every {int(uniform):,} training updates" if uniform else
                                                       'on its declared schedule') +
                             ", pausing training when a comparison needs its memory. Training updates are experience; "
                             "they are not skills."))
    next_step = pending[0] if pending else None
    control = read_json_or(saved/'supervisor_control.json', {})
    if open_window:
        done = len(window_tasks-in_window) if checkpoint is not None else 0
        active = tasks.get(str((checks or {}).get('active_task') or ''), {})
        activity.update(state='comparisons', done=done, total=len(window_tasks), unit='scheduled comparisons done in this window',
                        title=(f"Scheduled comparisons on saved step {checkpoint:,}" if checkpoint else
                               'Saving training for scheduled comparisons'))
        activity['status_label'] = ((f"Training is saved and paused while the comparisons use its memory: {done} of "
                                     f"{len(window_tasks)} done" + (f"; running now: {active['title']}" if active.get('status') == 'running'
                                                                 and active.get('title') else '') + '.')
                                    if window.get('state') == 'ready' else
                                    'Saving training before the scheduled comparisons; training resumes after them.')
    elif window.get('state') == 'held for review':
        # a mechanism that steers production failed its scheduled round: the scheduler keeps the saved
        # pause for review (ability_comparison_window.hold_for_review, the operator's rule of 2026-09-25)
        activity.update(state='attention', status_label=('Training is held for review: ' + str(window.get('reason')
                                                         or 'a scheduled comparison failed its declared line').rstrip('.') + '.'))
    elif not running and window.get('state') in ('repair required', 'ownership changed', 'control conflict'):
        # a comparison pause that did not end in a release: the scheduler withholds the resume
        activity.update(state='attention', status_label=('The scheduled-comparison pause ended in "' + str(window['state'])
                                                         + '": ' + str(window.get('reason') or 'no reason was recorded').rstrip('.') + '.'))
    elif launched and str(supervisor.get('status')) == 'training' and stamp is not None and 0 <= now-stamp < 600:
        activity.update(state='starting', status_label=(f"Training relaunched from saved step {step:,} at {clock(stamp)}; "
                                                        "its first training report has not arrived yet."))
    elif running and next_step is not None and next_step <= step:
        statuses = sorted({str(t.get('status')) for t in group(next_step) if t.get('status')})
        activity.update(state='due', status_label=(f"Scheduled comparisons were due at step {next_step:,}; the scheduler reports: "
                                                   + ', '.join(statuses) + '.' if statuses else
                                                   f"Scheduled comparisons were due at step {next_step:,}."))
    elif running:
        activity['state'] = 'running'
        if next_step is not None:
            span = int(next((every.get(k) for k, t in tasks.items() if _n(t.get('due_step')) == next_step and every.get(k)),
                            uniform or 3000))
            done = int(max(0, min(span, step-(next_step-span))))
            activity.update(start_step=next_step-span, done=done, total=span,
                            status_label=f"{done:,} of {span:,} training updates toward the next scheduled comparison at step {next_step:,}")
            if rate and rate > 0:
                activity['seconds_remaining'] = (next_step-step)/rate
                activity['estimate_basis'] = (f"Remaining updates to step {next_step:,} divided by the newest measured rate "
                                              f"({fixed(rate, 2)} model steps per second, recorded {clock(stamp)}).")
        else:
            activity['status_label'] = (f"Step {step:,}; no scheduled comparison is due." if tasks else
                                        f"Step {step:,}; the comparison scheduler's status file is missing, so no schedule is known.")
    elif str(control.get('cmd')) == 'pause' or supervisor.get('pause_reason'):
        reason = str(control.get('reason') or '').strip() if str(control.get('cmd')) == 'pause' else ''
        reason = reason or str(supervisor.get('pause_reason') or '').strip()
        activity.update(state='paused', status_label='Training paused' + (f": {reason.rstrip('.')}." if reason else '; progress saved.'))
    else:
        activity.update(state='stale', status_label=(f"Training report overdue: the newest record is {_age_words(now-stamp)} old."
                                                      if stamp is not None else 'Training report overdue.'))

    def card(due, base=None):
        # base: the pending step whose comparisons repeat at `due` (every task shares one due step)
        titles = [t.get('title') for t in group(due if base is None else base) if t.get('title')]
        name = (titles[0] if len(titles) == 1 else f"{len(titles)} scheduled comparisons") + f" at step {due:,}"
        detail = ("Runs automatically on the first complete saved checkpoint at or after that step"
                  + (': ' + '; '.join(titles) if len(titles) > 1 else '') + ". Each uses a frozen copy of the model in "
                  "all 8 saved worlds; training pauses when a comparison needs more memory than is free beside it. A "
                  "finished comparison records an outcome; it does not pass an ability or change a control switch.")
        # a due step already reached has no updates left to count: it is overdue, never "0 updates away"
        return name, detail, (due-reference if due > reference else None)

    def overdue(due):
        # reached and not completed (a completed task carries its next due step); the training state
        # comes from the supervisor's files, never from the scheduler's silence
        if due > reference:
            return None
        paused = str(control.get('cmd')) == 'pause' or bool(supervisor.get('pause_reason'))
        state = ('production training is paused' if paused else 'production training is running' if running else
                 'the training status is unavailable')
        return f"Overdue: due at step {due:,}; this check has not run; {state}."
    if checks_stamp is None or now-checks_stamp >= 180:
        # the due steps are the scheduler's own; nothing runs while it is not reporting
        activity['scheduler_overdue'] = True
        activity['scheduler_note'] = (f"The scheduler last reported {clock(checks_stamp)}; no comparison starts while it is not running."
                                      if checks_stamp is not None else 'The scheduler has not reported; no comparison starts until it runs.')
        activity['scheduler_status'] = (f"Scheduler status unavailable since {clock(checks_stamp)}" if checks_stamp is not None
                                        else 'Scheduler status unavailable')
    if next_step is not None:
        activity['next'], activity['next_detail'], activity['next_remaining_updates'] = card(next_step)
        activity['next_overdue'] = overdue(next_step)
        activity['next_step'] = next_step
        activity['next_due_at'] = None
        if activity['state'] == 'running' and rate and rate > 0 and next_step > step:
            remaining = next_step-step
            activity['next_due_at'] = stamp + remaining/rate
            activity['next_calculation'] = (f"{remaining:,} model steps ÷ {rate:.2f} model steps per second (newest record "
                                            f"{clock(stamp)}) = about {duration(remaining/rate)}. The comparison starts at the "
                                            "first complete saved checkpoint at or after that step.")
        if activity.get('scheduler_note'):
            activity['next_detail'] += ' ' + activity['scheduler_note']
        later = pending[1] if len(pending) > 1 else (next_step+int(uniform) if uniform else None)
        if later is not None:
            activity['then'], activity['then_detail'], activity['then_remaining_updates'] = \
                card(later, base=next_step if len(pending) < 2 else None)
            activity['then_step'] = later
            activity['then_overdue'] = overdue(later)
            # the comparison at next_step pauses training first when it needs the memory; that pause
            # varies, so no clock time is given for the later round
            activity['then_due_at'] = None
            activity['then_calculation'] = (f"No clock time: training may pause for the comparison at step {next_step:,} "
                                            "before reaching this step." if later > reference else
                                            f"Training already passed this step: its newest step is {reference:,}.")
    return activity


def project_records(root, now=None):
    """The header's Project records menu, each link with the time its file (or the data file its
    page reads) was last written, in AM/PM. The 2026-09-24 menu called a morning review the
    'Current investigation' and linked a build page whose writer had stopped the day before."""
    now = time.time() if now is None else now
    out = []
    for label, relative, data in PROJECT_RECORDS:
        path = root/relative
        if not path.is_file():
            continue
        source = root/data if data and (root/data).is_file() else path
        stamp = source.stat().st_mtime
        row = dict(label=label, url=file_url(root, path), updated_at=stamp, stale=False,
                   updated=(f"its data file last written {clock(stamp)}" if source != path else f"file last written {clock(stamp)}"))
        if source != path and source.suffix == '.json':
            # the build page's own rule (docs/build_status.html: staleLine = max(60, 4 x every_s)):
            # older data means its writer is not running
            every = _n(read_json_or(source, {}).get('every_s')) or 15
            if now-stamp > max(60, 4*every):
                row.update(stale=True, updated=f"its writer stopped; data last written {clock(stamp)}")
        out.append(row)
    return out


PROJECT_RECORDS = (
    ('Motor control review', 'docs/gain_of_function_v3/MOTOR_CONTROL_REVIEW_2026-09-23.md', None),
    ('Planner actions and triggers', 'docs/gain_of_function_v3/PLANNER_ACTIONS_2026-09-23.json', None),
    ('Hand actions and triggers', 'docs/gain_of_function_v3/HAND_PROGRAM_ACTIONS_2026-09-23.json', None),
    ('Scheduled comparisons: status and receipts', ABILITY_STATUS, None),
    ('Build progress', 'docs/build_status.html', 'Saved/mimo_bridge/build_status.json'),
    ('Takeover review', 'docs/gain_of_function_v3/TAKEOVER_REVIEW_2026-09-23.html', None),
    ('Old dashboard', 'Saved/mimo_bridge/dashboard_legacy.html', None),
)


def explainer_state(root, training):
    """When 'How the model works' (Saved/mimo_bridge/explainer.html) was built and from which run.
    The page is rebuilt only while its builder runs; the header says its age instead of implying it
    follows training."""
    # a second dashboard started with --explainer serves its own build at the same address (serve_beside)
    path = Path(EXPLAINER_PAGE) if EXPLAINER_PAGE else root/'Saved/mimo_bridge/explainer.html'
    if not path.is_file():
        return None
    stamp = path.stat().st_mtime
    try:
        with path.open('rb') as stream:
            head = stream.read(16_384).decode('utf-8', errors='replace')
    except OSError:
        head = ''
    match = re.search(r'<title>[^<]*?(H\d+[\w-]*)\s*</title>', head)
    run = match.group(1) if match else None
    active = (training or {}).get('run_name')
    label = f"built {clock(stamp)}" + (f" from {_segment(run)}" if run else '')
    detail = (f"Built {clock(stamp)}" + (f" from run {run}" if run else '') + '.'
              + (f" Training is now in {_segment(active)} ({active}); the page shows the model as it was when built."
                 if run and active and run != active else '')
              + ' It is rebuilt only while its builder runs.')
    return dict(url=file_url(root, root/'Saved/mimo_bridge/explainer.html'), built_at=stamp, run=run,
                current=bool(run and active and run == active), label=label, detail=detail)


def comparison_schedule(brief, results, activity, record, now, running):
    """The Now tab's test schedule from the scheduler, replacing training_brief's fixed "Not
    scheduled" sentence once the synergy-planning comparison runs automatically, and a timeline row
    for the next scheduled comparison."""
    from .training_brief import duration
    rows = {r['id']: r for r in (results or {}).get('rows', [])}
    body = rows.get('body_and_synergy')
    brief = dict(brief or {})
    if body:
        latest, every, due = body.get('latest') or {}, body.get('every_steps'), body.get('due_step')
        # the page's synergy test plan names planning AND exploration; say which one is scheduled
        exploration = any(re.search(r'explor|babble', f"{r.get('id')} {r.get('title')}", re.I) for r in rows.values())
        brief['test_status'] = (f"Scheduled: '{body.get('title') or 'Reachable body goals'}' compares the synergy planner with the starting controller "
                                + ((f"on a saved checkpoint when one of its declared triggers fires, and at least every {int(every):,} "
                                    "training updates" if body.get('cadence') == 'triggered' else
                                    f"on a saved checkpoint every {int(every):,} training updates") if every else 'on its declared schedule')
                                + (f"; latest at step {int(latest['step']):,} ({latest['when']})" if _n(latest.get('step')) is not None else '')
                                + (f"; next at step {int(due):,}" if due else '') + '.'
                                + ('' if exploration else ' No scheduled comparison covers synergy exploration.'))
        brief['test_scheduled'] = True
        if latest.get('seconds'):
            brief['test_duration'] = (f"Its latest round took {duration(latest['seconds'])} of computer time, measured from "
                                      "its receipt.")
    a = activity or {}
    if a.get('kind') == 'production' and a.get('next_step') is not None:
        stamp, rate, step = _n((record or {}).get('timestamp')), _n(a.get('rate')), _n(a.get('step'))
        fresh = bool(running) and stamp is not None and 0 <= now-stamp < 600
        due = a.get('next_due_at') if fresh else None
        remaining = max(0, a['next_step']-(step or 0))
        events = list(brief.get('events') or [])
        events.append(dict(
            title='Next scheduled physical comparison', due=due, kind='automatic',
            detail=a['next'] + '. A finished comparison records an outcome; it does not test a repeatable skill by itself.',
            calculation=((f"{remaining:,} model steps ÷ {rate:.2f} model steps per second = about {duration(remaining/rate)} "
                          "after the last report. The comparison starts at the first complete saved checkpoint at or after "
                          "that step.")
                         if rate and rate > 0 else 'The update rate is not recorded, so no time is calculated.'),
            status=('Overdue — completion not recorded' if fresh and _n(due) is not None and due < now else
                    'Estimated' if fresh and _n(due) is not None else 'Estimate unavailable while training is paused or unconfirmed')))
        events.sort(key=lambda e: e['due'] if _n(e.get('due')) is not None else float('inf'))
        brief['events'] = events
    return brief


def production_launches(root, live, saved_step):
    """Every launch on the active run's ancestry at or after `saved_step`, oldest first, as
    (launch time, run folder, launch record). The first is the resume past a repair stop."""
    launches = []
    for run in ancestry_bounds(root, dict(run=live['run'], step=live['step'])):
        record = _launch_record(Path(run)/'replay_training_stats.jsonl')
        if _n(record.get('step')) is not None and _n(record.get('timestamp')) is not None and record['step'] >= saved_step:
            launches.append((record['timestamp'], Path(run), record))
    return sorted(launches, key=lambda launch: launch[0])


_KEPT_CHECKS = (('initial_learning_state_preserved', 'learning state preserved'),
                ('eight_worlds_advanced_exactly', 'all eight worlds advanced exactly'),
                ('no_world_resets', 'no world reset'))


def resume_event(root, launches):
    """The repair history's missing entry: the first production launch past the repair stop, read
    from that run's launch record, its continuation receipt and the qualification the receipt names,
    with the number of restarts since."""
    if not launches:
        return None
    stamp, run, record = launches[0]
    receipt_path = run/'continuation_receipt.json'
    receipt = read_json_or(receipt_path, {}) if receipt_path.is_file() else {}
    try:
        proof = read_json(confined(root, receipt['proof'])) if receipt.get('proof') else {}
    except (KeyError, OSError, TypeError, ValueError):
        proof = {}
    checks = proof.get('checks') if isinstance(proof.get('checks'), dict) else {}
    kept = [words for key, words in _KEPT_CHECKS if checks.get(key) is True]
    last_stamp, last_run, last = launches[-1]
    qualified = (proof.get('passed') is True and _n(proof.get('completed_at')) is not None and _n(proof.get('steps')) is not None)
    event = dict(step=int(record['step']), phase='Resumed', title='Production resumed from the repair stop', derived=True,
                 impact=(f"Training resumed at {clock(stamp)} as {_segment(run.name)} from saved step {int(record['step']):,}"
                         + (f": {str(receipt['reason']).rstrip('.')}." if receipt.get('reason') else '.')
                         + (f" Its declared qualification passed at {clock(proof['completed_at'])} over "
                            f"{int(proof['steps']):,} private updates." if qualified else '')),
                 retained=('Qualification checks passed: ' + '; '.join(kept) + '.' if kept else
                           'The continuation receipt records the restored checkpoint and settings.'),
                 next=((f"Training has restarted {len(launches)-1} more time{'s' if len(launches) != 2 else ''} since, most "
                        f"recently at {clock(last_stamp)} from step {int(last['step']):,} ({_segment(last_run.name)}). "
                        if len(launches) > 1 else '')
                       + 'The scheduled comparisons test saved checkpoints as training continues.'),
                 source_available=receipt_path.is_file(),
                 source_url=file_url(root, receipt_path) if receipt_path.is_file() else None)
    return event


# the ten-item panel's items that a scheduled comparison now measures on saved production checkpoints
SCHEDULED_ITEMS = {5: 'body_and_synergy', 6: 'chosen_self_touch', 7: 'gaze_prediction'}


def link_scheduled(panel, results):
    """Give the upgrade cards the newest scheduled result of their comparison. A probe line such as
    'the current version has not been tested here' is replaced once a newer scheduled round exists."""
    rows = {r['id']: r for r in (results or {}).get('rows', [])}
    for item in (panel or {}).get('items', []):
        latest = (rows.get(SCHEDULED_ITEMS.get(item.get('item'))) or {}).get('latest') or {}
        if not latest.get('described') or _n(latest.get('step')) is None:
            continue
        note = (f"Newest scheduled comparison: checkpoint {int(latest['step']):,}, finished {latest['when']}. "
                + latest['headline'])
        old = str(item.get('test_scope') or '')
        probe = re.search(r'update ([\d,]+)', old)
        item['test_scope'] = note + (f" The probe reading below is from update {probe.group(1)}." if probe else
                                     (' ' + old if old else ''))
    return panel


def link_highlights(data, results):
    """A highlight tile that names a scheduled comparison ('scheduled': its task id) shows that
    comparison's newest value, checkpoint and time; its hand-written value and scope stay only until
    the first scheduled result exists (dashboard staleness check, 2026-09-24: the tiles still showed
    3:31 AM values after six scheduled rounds)."""
    rows = {r['id']: r for r in (results or {}).get('rows', [])}
    for tile in data.get('highlights', []):
        row = rows.get(tile.get('scheduled'))
        latest = (row or {}).get('latest') or {}
        if not row or not row.get('tile') or not latest.get('described') or _n(latest.get('step')) is None:
            continue
        tile.update(value=row['tile'], source_url=latest['url'], source_available=True, live=True,
                    scope=(f"Newest scheduled round: saved step {int(latest['step']):,}, finished {latest['when']}; "
                           f"{row['completed']} round{'s' if row['completed'] != 1 else ''} since step {row['first_step']:,}"
                           if _n(row.get('first_step')) is not None else
                           f"Newest scheduled round: saved step {int(latest['step']):,}, finished {latest['when']}"))
    return data


# ---------------------------------------------------------------- the capability programme's cards
# Operator, 2026-09-30: "The 'Build accurate, repeatable control' tab is completely out of date." Every
# card's measured numbers now come from its newest result file at page-build time; the hand-written
# text only explains. Each card carries 'checked_at' (when its text was last checked against its
# sources) and 'measured' (its bindings):
#   {"scheduled": <task id>, "show": headline|trend|detail|tile|next|limit|ranking|lead, "into": <field>}
#       a sentence the scheduled comparison's newest result already says in plain words;
#   {"file": <path>, "say": <text with {names}>, "values": {name: {"path": "a.b.0", ...}}, "time": <path>}
#       numbers read from one result file ("time": the file's own time field, else its write time);
#   {"training": true, "say": ..., "values": ...}
#       numbers from the active run's newest training record.
# A binding marked "live" (a report rewritten every few minutes, the training record) cannot date the
# text: its "assumes" list states what the text takes for granted, and a reading that breaks one marks
# the card out of date. Any other result newer than the card's checked_at marks it out of date.
CATALOG = 'docs/gain_of_function_v3/dashboard_goal_catalog.json'
CARD_GROUPS = ('highlights', 'capabilities', 'investigations', 'events', 'attention', 'hand_actions', 'upgrade_progress')
_SCHEDULED_SHOWS = ('headline', 'trend', 'detail', 'tile', 'next', 'limit', 'ranking', 'lead', 'cadence', 'tie', 'captured')


def _dig(data, path):
    """The value at a dotted path ('summary.best.0.x'), list indices as numbers; None when absent."""
    for part in str(path).split('.'):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list) and re.fullmatch(r'-?\d+', part) and -len(data) <= int(part) < len(data):
            data = data[int(part)]
        else:
            return None
    return data


def _number_text(value, spec):
    """One bound number in the page's plain form: 'percent' (a share, 1 = all), 'count' (a whole
    number with thousands separators) or significant digits (plain_numbers.fixed), then its unit."""
    from .plain_numbers import fixed, pct
    v = _n(value)
    if v is None:
        return None
    v = v * spec['scale'] if _n(spec.get('scale')) is not None else v
    if spec.get('absolute'):
        v = abs(v)
    digits = int(spec['digits']) if _n(spec.get('digits')) is not None else 2
    text = pct(v, digits) if spec.get('percent') else f"{int(round(v)):,}" if spec.get('count') else fixed(v, digits)
    return text + (f" {spec['unit']}" if spec.get('unit') else '')


def _stamp_of(value):
    """A time written in a result file: a Unix time, or a local '2026-09-24 19:15:11' text."""
    if _n(value) is not None and value > 1e9:
        return float(value)
    return local_stamp(value) if isinstance(value, str) else None


def _assumption_broken(data, rule):
    """The 'means' sentence of an assumption the reading breaks, else None. A true/false field is
    checked with 'equals'."""
    if not isinstance(rule, dict):
        return f"an assumption is not written as a rule ({rule!r})"
    raw = _dig(data, rule.get('path'))
    if isinstance(raw, bool) or isinstance(rule.get('equals'), bool):
        return None if raw is rule.get('equals') else (rule.get('means') or f"{rule.get('path')} is {rule.get('equals')}")
    v = _n(raw)
    if v is None:
        return rule.get('means') or f"{rule.get('path')} has a reading"
    checks = (('equals', lambda a, b: a == b), ('below', lambda a, b: a < b), ('above', lambda a, b: a > b),
              ('at_most', lambda a, b: a <= b), ('at_least', lambda a, b: a >= b))
    for key, holds in checks:
        if key in rule and _n(rule[key]) is not None and not holds(v, rule[key]):
            return rule.get('means') or f"{rule.get('path')} {key} {rule[key]}"
    return None


def _say(data, binding):
    """The binding's sentence with its numbers filled from `data`, or (None, missing names)."""
    values, missing = {}, []
    for name, spec in (binding.get('values') or {}).items():
        spec = spec if isinstance(spec, dict) else dict(path=spec)
        raw = _dig(data, spec.get('path'))
        if spec.get('over') is not None:
            below = _n(_dig(data, spec['over']))
            raw = raw / below if _n(raw) is not None and below else None
        text = raw if isinstance(raw, str) and spec.get('text') else _number_text(raw, spec)
        if text is None:
            missing.append(spec.get('path') or name)
        values[name] = text
    if missing:
        return None, missing
    return str(binding.get('say') or '').format(**values), []


def _active_training_record(root):
    supervisor = read_json_or(root/'Saved/mimo_bridge/training_continuation.json', {})
    try:
        run = confined(root, supervisor['active_run']) if supervisor.get('active_run') else None
    except (TypeError, ValueError, OSError):
        run = None
    return (last_training_record(run/'replay_training_stats.jsonl') if run else {}), run


def _bound(root, binding, rows, record):
    """One binding's sentence and the time, step and file of the result it came from."""
    into = binding.get('into') or 'evidence'
    out = dict(into=into, text=None, at=None, step=None, url=None, live=bool(binding.get('live')), missing=None,
               broken=None, label=None)
    if binding.get('scheduled'):
        task, show = binding['scheduled'], binding.get('show') or 'headline'
        row = rows.get(task)
        latest = (row or {}).get('latest') or {}
        text = (row or {}).get(show) if show in ('trend', 'tile', 'next') else latest.get(show)
        if show == 'tile' and not text:
            text = latest.get('tile')
        if show == 'cadence' and row and _n(row.get('every_steps')) is not None:
            every = int(row['every_steps'])
            text = (f"It runs when one of its declared triggers fires and at least every {every:,} training updates."
                    if row.get('cadence') == 'triggered' else f"It runs every {every:,} training updates.")
        if not row or not latest or not text:
            out['missing'] = (f"the scheduled comparison '{(row or {}).get('title') or task}' has no completed result "
                              f"that says this ({show})")
            return out
        name = str(row.get('title') or task).split(':')[0]
        out.update(text=text, at=_n(latest.get('completed_at')), step=_n(latest.get('step')), url=latest.get('url'),
                   label=f"'{name}' on saved step {int(latest['step']):,} (finished {latest['when']})"
                   if _n(latest.get('step')) is not None else f"'{name}' (finished {latest.get('when')})")
        return out
    if binding.get('training'):
        data, run = record
        if not data:
            out['missing'] = 'the active training run has no readable training record'
            return out
        text, missing = _say(data, binding)
        if missing:
            out['missing'] = 'the newest training record has no ' + ', '.join(missing)
            return out
        out.update(text=text, at=_n(data.get('timestamp')), step=_n(data.get('step')), live=True,
                   url=file_url(root, run/'replay_training_stats.jsonl') if run else None,
                   label=f"the training record of step {int(data['step']):,} ({clock(data['timestamp'])})")
        out['broken'] = next((b for b in (_assumption_broken(data, r) for r in binding.get('assumes') or []) if b), None)
        return out
    path_text = binding.get('file')
    try:
        path = confined(root, path_text)
    except (TypeError, ValueError, OSError):
        out['missing'] = f"{path_text} is outside the project or not named"
        return out
    data = _result_payload(path) if path.suffix == '.json' else None
    if not path.is_file() or (path.suffix == '.json' and not isinstance(data, (dict, list))):
        out['missing'] = f"{path_text} is missing or unreadable"
        return out
    text, missing = _say(data, binding)
    if missing:
        out['missing'] = f"{path_text} has no " + ', '.join(missing)
        return out
    at = _stamp_of(_dig(data, binding['time'])) if binding.get('time') else None
    at = at if at is not None else path.stat().st_mtime
    out.update(text=text, at=at, url=file_url(root, path),
               label=f"{binding.get('name') or path.name} ({clock(at)})")
    out['broken'] = next((b for b in (_assumption_broken(data, r) for r in binding.get('assumes') or []) if b), None)
    return out


def _fill(card, field, texts):
    """Put measured sentences into a card field: 'value' is replaced, 'stage_evidence.N' fills one
    stage's sentence, any other field gets them after its explanatory text."""
    if not texts:
        return
    joined = ' '.join(t.strip() for t in texts if t)
    if field == 'value':
        card['value'] = joined
        return
    if field.startswith('stage_evidence.'):
        index = int(field.split('.', 1)[1])
        stages = list(card.get('stage_evidence') or [])
        while len(stages) <= index:
            stages.append('')
        stages[index] = (stages[index] + ' ' + joined).strip()
        card['stage_evidence'] = stages
        return
    card[field] = (str(card.get(field) or '') + ' ' + joined).strip()


def bind_card(root, card, rows, record, prefer_result=False):
    """Fill one card's numbers from its bindings and date it: when its text was checked, which
    results its numbers come from, and since when it is out of date. The card's own source document
    (its explanation's source) dates it too. prefer_result: link the newest result file instead of
    the source document (the highlight tiles)."""
    checked = local_stamp(card.get('checked_at'))
    bound = [_bound(root, b, rows, record) for b in card.get('measured') or [] if isinstance(b, dict)]
    by_field = {}
    for b in bound:
        by_field.setdefault(b['into'], []).append(b['text'] if not b['missing'] else
                                                  f"(Missing: {b['missing']}; no number is shown.)")
    for field, texts in by_field.items():
        _fill(card, field, texts)
    firsts = [b for b in bound if b['url']]
    if firsts and (prefer_result or not card.get('source_available')):
        card.update(source_url=firsts[0]['url'], source_available=True)
    dated = [b for b in bound if not b['missing'] and not b['live'] and b['at'] is not None]
    document = []
    if card.get('source_available') and card.get('source'):
        try:
            path = confined(root, card['source'])
            if all(b['url'] != file_url(root, path) for b in bound):
                stamp = path.stat().st_mtime
                document = [dict(at=stamp, label=f"{path.name} (written {clock(stamp)})", missing=None)]
        except (OSError, ValueError):
            document = []
    dated += document
    newest = max((b['at'] for b in dated), default=None)
    broken = [b for b in bound if b.get('broken')]
    labels = []
    for b in bound + document:
        if b['label'] and b['label'] not in labels and not b['missing']:
            labels.append(b['label'])
    fresh = dict(checked_at=checked, checked_text=clock(checked) if checked else None, newest_at=newest,
                 missing=[b['missing'] for b in bound if b['missing']], sources=labels,
                 out_of_date=False, since=None, since_text=None)
    text = (f"Text checked {clock(checked)}" if checked else 'Text has no checked date') + (
        f"; read from {'; '.join(labels)}" if labels else '') + '.'
    if fresh['missing']:
        text += f" Missing: {'; '.join(fresh['missing'])}."
    if checked is None:
        fresh.update(out_of_date=True, stale_reason='this card does not say when its text was checked.')
    elif newest is not None and newest > checked:
        fresh.update(out_of_date=True, since=newest, since_text=clock(newest),
                     stale_reason=(f"a newer result arrived after this text was checked ({clock(checked)}). The numbers "
                                   "shown come from the newest results; the explanation has not been re-checked "
                                   "against them."))
    elif broken:
        first = min(broken, key=lambda b: b['at'] or 0)
        fresh.update(out_of_date=True, since=first['at'], since_text=clock(first['at']),
                     stale_reason=f"the text assumes that {first['broken']}, and the newest reading no longer shows that.")
    if fresh['out_of_date']:
        fresh['stale_text'] = ('Out of date' + (f" since {fresh['since_text']}" if fresh['since_text'] else '') + ': '
                               + fresh['stale_reason'])
    fresh['text'] = text
    card['freshness'] = fresh
    return card


def bind_cards(root, data, results):
    """Bind every card; a malformed card is shown as out of date with its error, never the whole tab
    lost (goal_catalog runs inside the page's guard, where one exception would blank the tab)."""
    rows = {r['id']: r for r in (results or {}).get('rows', [])}
    record = _active_training_record(root)
    for group in CARD_GROUPS:
        for card in data.get(group, []):
            if isinstance(card, dict) and not card.get('derived'):
                try:
                    bind_card(root, card, rows, record, prefer_result=(group == 'highlights'))
                except Exception as exc:   # noqa: BLE001 - shown on the card
                    why = f"its bindings could not be read ({type(exc).__name__}: {exc})"
                    card['freshness'] = dict(out_of_date=True, since=None, since_text=None, missing=[why],
                                             checked_at=None, checked_text=None, sources=[], stale_reason=why + '.',
                                             stale_text='Out of date: ' + why + '.', text='Missing: ' + why + '.')
    return data


# a fixed test definition (a horizon, a goal, a simulator joint range) may keep its digits in the
# hand-written text; so may names of places in training and time (steps, segments, dates, clock times,
# dated file names, commits) and the ends of a scale ("0% = no better, 100% = exact"). Any other digit
# in hand-written text is a typed measurement.
_IDENTITY_NUMBERS = re.compile(
    r"(?i:\b(?:steps?|checkpoints?|segments?|seg)\s+[\d,]+(?:\s*(?:to|-)\s*[\d,]+)?)"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|June|July|August"
    r"|September|October|November|December)\.?\s+\d{1,2}\b(?:,\s*\d{4})?"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)\b"
    r"|[\w./\\-]*\d{4}-\d{2}-\d{2}[\w./\\-]*"
    r"|(?=[\w.:/\\-]*[A-Za-z_])[\w.:-]*[/\\][\w./\\:-]+"
    r"|\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b"
    r"|--[\w-]+"
    r"|\{[\w]+\}"
    r"|\b(?:0|1|50|100)(?:%|\s+times?)?(?:\s+less(?:\s+error)?)?\s*=")
_CARD_KEYS_NOT_TEXT = {'id', 'source', 'checked_at', 'tone', 'scheduled', 'stage_states', 'upgrades', 'step', 'item',
                       'path', 'over', 'file', 'time', 'show', 'into', 'training', 'live', 'percent', 'count', 'digits',
                       'scale', 'absolute', 'equals', 'below', 'above', 'at_most', 'at_least'}


def typed_numbers(card, fixed_definitions=()):
    """Every digit-bearing stretch of a card's hand-written text that is neither a fixed test
    definition nor a place in training or time: (field, the text around it)."""
    hits = []

    def walk(value, key):
        if isinstance(value, dict):
            for k, v in value.items():
                if k not in _CARD_KEYS_NOT_TEXT:
                    walk(v, k)
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str):
            text = value
            for phrase in sorted(fixed_definitions, key=len, reverse=True):
                text = text.replace(phrase, ' ')
            text = _IDENTITY_NUMBERS.sub(' ', text)
            for m in re.finditer(r'\d', text):
                hits.append((key, text[max(0, m.start() - 30):m.end() + 30].strip()))
                break
    walk(card, None)
    return hits


_RESTART_FLAG = re.compile(r'[\\/]')


def _receipt_time(text):
    """'Sep 29, 2026, 06:00:33 AM PDT' (the coordinator's receipts) as a Unix time, else None."""
    match = re.match(r'\s*([A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}(?::\d{2})? [AP]M)', str(text or ''))
    for layout in ('%b %d, %Y, %I:%M:%S %p', '%b %d, %Y, %I:%M %p'):
        try:
            return datetime.strptime(match.group(1), layout).timestamp() if match else None
        except ValueError:
            continue
    return None


def restart_events(root, receipt_path, title='Restarts'):
    """One derived history entry for every restart a segment's launch receipt records: from which
    saved step, with which setting changed, why, and whether its first training record reinitialized
    anything. The receipt is appended at every restart, so the history follows it without hand edits."""
    try:
        path = confined(root, receipt_path)
    except (TypeError, ValueError, OSError):
        return None
    receipt = read_json_or(path, None)
    if not isinstance(receipt, dict):
        return None
    events, current = [], None
    for e in receipt.get('events') or []:
        if not isinstance(e, dict):
            continue
        kind = str(e.get('event') or '')
        if kind == 'relaunch: old supervisor ended':
            match = re.search(r'ckpt_step_(\d+)', str(e.get('checkpoint') or ''))
            current = dict(at=e.get('at'), step=int(match.group(1)) if match else None, reason=e.get('reason'),
                           changes=e.get('changes') or {}, reinitialized=None, run=None)
            events.append(current)
        elif current is not None and kind == 'relaunch: trainer launched':
            current['run'] = Path(str(e.get('run') or '')).name or None
        elif current is not None and kind == 'relaunch: first stats record':
            current['reinitialized'] = e.get('reinitialized')
    if not events:
        return None
    clean = [r for r in events if r['reinitialized'] == []]
    refused = sum(1 for e in receipt.get('events') or [] if isinstance(e, dict)
                  and e.get('event') == 'relaunch: checkpoint refused')
    fixes = []
    for r in events:
        flags = list(r['changes']) if isinstance(r['changes'], dict) else []
        changed = (f"{' and '.join(flags)} changed" if flags else '')
        r['when'] = clock(_receipt_time(r['at'])) if _receipt_time(r['at']) else 'At an unrecorded time'
        fixes.append(dict(name=f"{r['when']}, from saved step "
                               + (f"{r['step']:,}" if r['step'] is not None else 'unrecorded')
                               + (f", with {changed}" if changed else ', no setting changed'),
                          impact='Recorded reason: ' + am_pm_text(str(r['reason'] or 'none recorded.'))))
    newest = events[-1]
    stamp = path.stat().st_mtime
    where = f" from step {newest['step']:,}" if newest['step'] is not None else ''
    return dict(step=newest['step'], phase='Restarted', derived=True,
                title=f"{title}: {len(events)} so far",
                impact=(f"After each stop training was relaunched from the newest complete saved checkpoint. The newest "
                        f"relaunch, {newest['when']}, resumed{where}. The reason recorded for each relaunch is listed "
                        "in its details."),
                retained=(f"{len(clean)} of {len(events)} relaunches reported in their first training record that no "
                          "learned part was reinitialized. "
                          + (f"The relaunch tool refused an incomplete checkpoint {refused} time{'s' if refused != 1 else ''} "
                             "and used the newest complete one. " if refused else '')
                          + "The supervisor's own resumes after scheduled-comparison pauses are not listed."),
                next='Each relaunch, its setting change and its reason are listed above, oldest first.',
                fixes=fixes, source_available=True, source_url=file_url(root, path),
                freshness=dict(text=f"Read from the launch receipt, last written {clock(stamp)}.", out_of_date=False,
                               checked_at=stamp, checked_text=clock(stamp), sources=[path.name], missing=[]))


_SUPERSEDED_STATE = {'complete': 'finished', 'stopped': 'stopped', 'stale': 'no longer reporting',
                     'waiting': 'waiting for a report that never came'}


def goal_catalog(root, checks=None, now=None, results=None, catalog_path=None):
    """Dated research decisions, separate from live process measurements. catalog_path: a catalogue
    file other than root's (a worktree's, checked against the live data root)."""
    path = Path(catalog_path) if catalog_path else root / CATALOG
    if not path.is_file():
        return None
    data = read_json(path)
    if data.get('schema') != 1:
        raise ValueError('Unsupported goal catalogue schema')
    states = {'signal', 'concern', 'untested', 'passed'}
    if len(data.get('stages', [])) != 4:
        raise ValueError('The goal catalogue must define all four proof stages')
    for item in data.get('capabilities', []):
        if len(item.get('stage_states', [])) != 4 or not set(item['stage_states']) <= states:
            raise ValueError('Invalid capability proof stages')
    for group in ('hand_actions', 'upgrade_progress'):
        for item in data.get(group, []):
            if len(item.get('stage_states', [])) != 4 or not set(item['stage_states']) <= states:
                raise ValueError('Invalid detailed capability proof stages')
    for group in ('highlights', 'capabilities', 'investigations', 'events', 'attention', 'hand_actions', 'upgrade_progress'):
        for item in data.get(group, []):
            source = item.get('source')
            item['source_available'] = bool(source and confined(root, source).is_file())
            item['source_url'] = file_url(root, source) if item['source_available'] else None
    data['source_url'] = file_url(root, data['source'])
    hand_program = root / 'docs/gain_of_function_v3/HAND_CAPABILITY_PROGRAM_2026-09-23.md'
    data['hand_program_url'] = file_url(root, hand_program) if hand_program.is_file() else None
    results = results if results is not None else ability_results(root, checks)
    link_highlights(data, results)
    bind_cards(root, data, results)
    spec = data.get('restarts') if isinstance(data.get('restarts'), dict) else {}
    try:
        derived = restart_events(root, spec['receipt'], spec.get('title') or 'Restarts') if spec.get('receipt') else None
    except Exception as exc:   # noqa: BLE001 - a malformed receipt costs only its history entry
        print(f'Dashboard: relaunch history unavailable: {type(exc).__name__}: {exc}', flush=True)
        derived = None
    if derived:
        data['events'] = list(data.get('events') or []) + [derived]
    # when these hand-written notes were last written (reviewed_at is a date only), and whether
    # production has moved past the state they describe (dashboard staleness check, 2026-09-24)
    written = path.stat().st_mtime
    data['written_at'], data['written_text'] = written, clock(written)
    activity = data.get('activity')
    live = production_activity(root, checks, now)
    if activity and activity.get('kind') == 'repair' and live:
        # a repair activity holds production saved at one step (its own start_step); a production
        # record at a later step means training moved past that state, and the card would present a
        # finished repair as current
        saved_step = _n(activity.get('saved_step', activity.get('start_step')))
        if saved_step is not None and live['step'] > saved_step:
            launches = production_launches(root, live, saved_step)
            data['superseded_activity'] = dict(title=activity.get('title'), saved_step=int(saved_step), written_at=written,
                                               resumed_at=launches[0][0] if launches else None)
            activity = data['activity'] = live
            event = resume_event(root, launches)
            if event:
                data['events'] = list(data.get('events') or []) + [event]
    elif not activity and live:
        activity = data['activity'] = live
    if activity and activity.get('kind') == 'production':
        pass                       # built complete by production_activity
    elif activity and activity.get('kind') == 'training':
        supervisor = read_json_or(root/'Saved/mimo_bridge/training_continuation.json', {})
        run = confined(root, supervisor.get('active_run') or activity['run'])
        record = last_training_record(run/'replay_training_stats.jsonl')
        total = activity['target_step']-activity['start_step']
        done = max(0, min(total, (record.get('step') or activity['start_step'])-activity['start_step']))
        fresh = bool(record and time.time()-record['timestamp'] < 600)
        control = read_json_or(root/'Saved/mimo_bridge/supervisor_control.json', {})
        # a repair run trains under a pause that names it (the relaunch review of Oct 5, 2026, finding F10): with its
        # stats fresh it is running, not paused
        repair = bool(control.get('repair_run')) and Path(str(control['repair_run'])).resolve() == Path(run).resolve()
        paused = control.get('cmd') == 'pause' and not (repair and fresh)
        activity.update(done=done,total=total,recorded_at=record.get('timestamp'),
                        state='milestone_due' if done>=total else 'paused' if paused else 'running' if fresh else 'stale',
                        seconds_remaining=None,unit='training updates toward the recheck')
        rate = record.get('model_steps_per_s')
        if fresh and not paused and done<total and isinstance(rate,(int,float)) and rate>0:
            activity['seconds_remaining']=(total-done)/rate
            activity['estimate_basis']='Remaining production updates divided by the latest measured rate. The checkpoint save and physical recheck follow.'
    elif activity and activity.get('kind') == 'repair':
        receipt = read_json_or(root/'Saved/gof_takeover_20260924/repair/source.json', {})
        activity.update(state='repairing', done=None, total=None, seconds_remaining=None,
                        recorded_at=None, unit='repairs and matched ability tests',
                        saved_step=receipt.get('step', activity.get('start_step')))
        if activity.get('work_status_path'):
            progress=read_json_or(confined(root,activity['work_status_path']),{})
            done,total=progress.get('completed'),progress.get('total')
            elapsed=progress.get('elapsed_s')
            timestamp=progress.get('timestamp')
            if timestamp is None and isinstance(progress.get('started_at'),(int,float)) and isinstance(elapsed,(int,float)):
                timestamp=progress['started_at']+elapsed
            fresh=isinstance(timestamp,(int,float)) and time.time()-timestamp<600
            if isinstance(done,int) and isinstance(total,int) and 0<=done<=total and total>0:
                activity.update(done=done,total=total,recorded_at=timestamp,
                    unit=activity.get('progress_unit','physical comparisons completed'))
                activity['work_status_label']=f"{done} of {total} "+activity['unit']
                if progress.get('state')=='running' and fresh:
                    activity['state']='running'
                    if done>0 and isinstance(elapsed,(int,float)) and elapsed>0:
                        activity['seconds_remaining']=(total-done)*elapsed/done
                        activity['estimate_basis']='Remaining comparisons multiplied by elapsed time per completed comparison. Finding the learner’s next goal can change the pace; review follows.'
                elif not fresh:
                    activity['work_status_label']+='; progress report overdue'
                else:
                    activity['work_status_label']+='; outcome under review'
    elif activity and activity.get('kind') == 'fixture':
        run = confined(root, activity['run'])
        record = read_json_or(run/'status.json', {})
        result = read_json_or(run/'report.json', {})
        declaration = run/'declaration.json'
        started = declaration.stat().st_mtime if declaration.is_file() else None
        timestamp = record.get('timestamp', started)
        fresh = isinstance(timestamp,(int,float)) and time.time()-timestamp < 600
        stopped = str(record.get('stage','')).startswith('stopped')
        done = sum((run/(name+'.json')).is_file() for name in activity['trial_files'])
        total = len(activity['trial_files'])
        activity.update(done=done,total=total,recorded_at=timestamp,
                        state='stopped' if stopped else 'complete' if result else 'running' if record and fresh else 'starting' if fresh else 'stale' if record else 'waiting',
                        seconds_remaining=None,unit='grasp/release comparisons recorded',
                        result_note=record.get('reason','Fixture stopped for correction.') if stopped else 'Physical fixture finished; its results are under review.' if result else None)
        if fresh and not stopped and not result and done and started is not None and timestamp > started:
            activity['seconds_remaining'] = (total-done)*(timestamp-started)/done
            activity['estimate_basis'] = 'Remaining comparisons multiplied by elapsed wall time per recorded comparison, including fixture setup so far. Review follows.'
    elif activity and activity.get('kind') == 'physical':
        record = read_json_or(confined(root, activity['status_path']), {})
        timestamp = record.get('timestamp')
        fresh = isinstance(timestamp,(int,float)) and time.time()-timestamp < 600
        finished = record.get('stage') == 'complete'
        attained = record.get('attained', record.get('physical_roll_found'))
        activity.update(done=record.get('completed_candidates',0),total=record.get('maximum_candidates',96),
                        recorded_at=timestamp,unit=activity.get('progress_unit','movement sequences checked'),
                        state='complete' if finished else 'running' if fresh else 'stale' if record else 'waiting',
                        passed=attained if finished else None,seconds_remaining=None,
                        result_note=('Supported roll found in the physical test.' if attained else
                                     'Search finished; the supported-roll goal was not reached.') if finished else None)
        elapsed = record.get('elapsed_s')
        started = record.get('started_at')
        if started is None and activity.get('timing_origin'):
            started = read_json_or(confined(root, activity['timing_origin']), {}).get('started_at')
        if elapsed is None and isinstance(started, (int, float)) and isinstance(timestamp, (int, float)):
            elapsed = timestamp-started
        if fresh and not finished and activity['done'] > 0 and isinstance(elapsed, (int, float)) and elapsed > 0:
            activity['seconds_remaining'] = max(0, activity['total']-activity['done'])*elapsed/activity['done']
            activity['estimate_basis'] = 'Remaining sequences multiplied by elapsed wall time per completed sequence. This estimates the search only; final replay and review follow. Early success can finish sooner.'
    elif activity:
        run = confined(root, activity['run'])
        record = last_training_record(run/'replay_training_stats.jsonl')
        result = read_json_or(confined(root, activity['result']), {})
        total = activity['updates']
        done = max(0, min(total, (record.get('step') or activity['start_step'])-activity['start_step']))
        fresh = bool(record and time.time()-record['timestamp'] < 600)
        declaration = run.parents[1]/'declaration.json'
        starting = declaration.is_file() and time.time()-declaration.stat().st_mtime < 600
        activity.update(done=done, total=total, recorded_at=record.get('timestamp'),
                        state='complete' if result else 'running' if fresh else 'stale' if record else 'starting' if starting else 'waiting',
                        passed=result.get('passed'), seconds_remaining=None,unit='test updates recorded')
        rate = record.get('model_steps_per_s')
        if fresh and not result and isinstance(rate,(int,float)) and rate > 0:
            activity['seconds_remaining'] = (total-done)/rate
            activity['estimate_basis'] = 'Remaining updates divided by the latest measured update rate; saving and review follow.'
        activity['result_note'] = ('Repair qualification passed.' if activity.get('kind')=='qualification' and result.get('passed') is True else
                                  'Experiment finished; qualification is a separate test.' if result and activity.get('kind')=='experiment' else
                                  'Qualification found unresolved checks.' if result else None)
    if (live and activity and activity is not live and activity.get('kind') in ('experiment', 'qualification', 'fixture', 'physical')
            and activity.get('state') in _SUPERSEDED_STATE):
        # a private test that finished, stopped or stopped reporting is no longer the present work;
        # production's own state takes the card and the test's words become dated history
        data['superseded_activity'] = dict(title=activity.get('title'), state=activity.get('state'), written_at=written)
        activity = data['activity'] = live
    if activity and activity.get('next_timing_run'):
        previous = last_training_record(confined(root, activity['next_timing_run'])/'replay_training_stats.jsonl')
        rate = previous.get('model_steps_per_s')
        if isinstance(rate, (int,float)) and rate > 0:
            activity['next_seconds_estimate'] = activity['next_updates']/rate
            activity['next_estimate_basis'] = 'Estimate: '+str(activity['next_updates'])+' updates divided by the preceding test’s measured rate. Setup, saving and review are additional; concurrent tests may change the rate.'
    if activity and activity.get('next_timing_result'):
        previous=read_json_or(confined(root,activity['next_timing_result']),{})
        elapsed,steps=previous.get('elapsed_s'),previous.get('steps')
        if isinstance(elapsed,(int,float)) and elapsed>0 and isinstance(steps,(int,float)) and steps>0:
            activity['next_seconds_estimate']=elapsed*activity['next_updates']/steps
            activity['next_estimate_includes_setup']=True
            activity['next_estimate_basis']='Estimate from the preceding complete private run: '+str(steps)+' updates took '+str(round(elapsed))+' seconds, including setup and saving. Scaled to '+str(activity['next_updates'])+' updates. Review follows; added checks or concurrent work can change the duration.'
    if activity and data.get('recovery'):
        recovery = dict(data['recovery'])
        supervisor = read_json_or(root/'Saved/mimo_bridge/training_continuation.json', {})
        if supervisor.get('active_run'):
            latest = last_training_record(confined(root, supervisor['active_run'])/'replay_training_stats.jsonl')
            step, rate = latest.get('step'), latest.get('model_steps_per_s')
            if isinstance(step, (int,float)):
                recovery['remaining_updates'] = max(0, recovery['step']-step)
                recovery['observed_step'] = step
                recovery['rate_recorded_at'] = latest.get('timestamp')
                if isinstance(rate, (int,float)) and rate > 0:
                    recovery['seconds_estimate'] = recovery['remaining_updates']/rate
        activity['recovery'] = recovery
    if data.get('superseded_activity'):
        old = data['superseded_activity']
        newest = f"its newest record is lineage step {live['step']:,} ({clock(live['recorded_at'])})"
        if old.get('saved_step') is None:
            data['as_of'] = f"Notes written {clock(written)}"
            data['staleness'] = (f"Written {clock(written)}; its Now card named '{old.get('title') or 'a private test'}', "
                                 f"which is {_SUPERSEDED_STATE[old['state']]}. Production training is the present work: "
                                 f"{newest}. Notes that describe that test as current describe that earlier state.")
        else:
            resumed = old.get('resumed_at')
            before = resumed is not None and written < resumed
            # the file's write time says when the notes were last saved; "before production resumed"
            # is said only when that time precedes the first launch past the saved step
            data['as_of'] = (f"Notes written {clock(written)}, before production resumed at {clock(resumed)}" if before
                             else f"Notes written {clock(written)}")
            data['staleness'] = ((f"Written {clock(written)}, while production was saved at step {old['saved_step']:,} for repairs. "
                                  + (f"Training resumed at {clock(resumed)}; {newest}. " if resumed is not None else
                                     f"Training has moved past that step; {newest}. "))
                                 if before or resumed is None else
                                 (f"Written {clock(written)}, but its Now card still described production saved at step "
                                  f"{old['saved_step']:,}, which training passed at {clock(resumed)}; {newest}. ")) + \
                "Notes that say production is paused describe that earlier state."
    return data


def build_state(root=REPO, registry_path=None, now=None):
    from .experiment_board import build as work_board
    from .developmental_status import build as developmental_status
    from . import lineage_series
    now = time.time() if now is None else now
    training = training_state(root, now)
    record, args = training.pop('_record', None) or {}, training.pop('_args', None) or {}
    bounds = ancestry_bounds(root, training)

    def guarded(what, fn, fallback):
        # the gain-of-function parts read fields other agents are adding right now; a malformed
        # record must cost only its own panel, never the whole page
        try:
            return fn()
        except Exception as exc:   # noqa: BLE001 - reported on the page, not swallowed
            print(f'Dashboard: {what} unavailable: {type(exc).__name__}: {exc}', flush=True)
            return fallback(f'{what} could not be computed ({type(exc).__name__}: {exc})')
    probes = guarded('the probe index', lambda: lineage_series.probe_results(root, bounds),
                     lambda why: dict(items={}, index_found=False, other_lineage=0, error=why))
    panel = guarded('the ten-item panel', lambda: lineage_series.gain_of_function(record, args, probes), lambda why: None)
    # built from the flag states, not from what is measured (reviews/dashboards.md M2): a
    # stand-in reading must not drop the flag that measures its item
    def with_value(flag):
        value = lineage_series.BLOCK_FLAGS.get(flag[2:].replace('-', '_'), 'on')
        return flag if value in ('on', '>0') else f'{flag} {value}'      # --wellness-rank normalized
    off = [with_value(row['flag']) for row in (panel or {}).get('items', []) if row.get('flag')
           and not str(row.get('flag_state','')).startswith('on')]
    next_step = (('Turning on '+', '.join(off)+' makes '+('its item' if len(off)==1 else 'those items')+
                  ' measurable; each flag defaults to the old behaviour, so a resume without them changes nothing.')
                 if off else 'Every item\'s flag is on; the probe items need the probe suite to run on this '
                             'lineage\'s checkpoints.') if panel else None
    verdict = guarded('the training verdict', lambda: training_verdict(root, training, now),
                      lambda why: dict(state='unconfirmed', sentence='Training state is not confirmed: '+why+'.', facts=[], source=None))
    if training.get('view') != 'experiment' and not production_owns_the_page(training, now):
        # an experiment that ran outside the supervisor and is not live now: named on the card, its clips stay listed
        note = guarded('the ended experiment', lambda: ended_experiment_note(root), lambda why: None)
        if note:
            verdict['facts'] = list(verdict.get('facts') or []) + [note]
    action_use = guarded('the action-use spread',
                         lambda: lineage_series.action_use_summary(root, bounds, cache_dir=LINEAGE_CACHE_DIR),
                         lambda why: None)
    headline = guarded('the headline', lambda: lineage_series.headline(panel, record, action_use) if panel else None, lambda why: None)
    changed = guarded('the newest segment', lambda: experiment_changed(root, training)
                      if training.get('view') == 'experiment' else what_changed(root, bounds), lambda why: why)
    liveness = guarded('the liveness card', lambda: liveness_card(record), lambda why: dict(title='Liveness unavailable', lines=[dict(kind='note', text=why)]))
    live = live_state(root, training, record, args)
    observations = guarded('the body observations', lambda: observation_cards(training, record, live), lambda why: [])
    registry = read_json(registry_path or root/'docs/development_dashboard/evidence.json')
    if registry.get('schema') != 1:
        raise ValueError('Unsupported evidence schema')
    records = evidence_records(root, registry, bounds)+reviewed_behavior_records(root,bounds)
    abilities = []
    for definition in registry['abilities']:
        item = dict(definition)
        matching = [r for r in records if r['capability']==item['id']
                    and r['kind']!='implementation'
                    and (r['kind']=='offline_component' or r['lineage']=='ancestor_checkpoint')]
        item['evidence'] = matching[-1] if matching else None
        abilities.append(item)
    from .training_brief import work_timeline
    work=work_board(root,now,bounds)
    automatic={r['id']:r for r in records if r.get('valid') and r.get('validation')=='automated_validated_observation'}
    for job in work.get('jobs',[]):
        if job['id'] in automatic and job.get('state')=='completed_unreviewed':
            observation=automatic[job['id']]
            job.update(state='measured',result=observation['summary'],
                limitation='Automatic replay- and identity-checked observation. '+observation['missing'])
    # the earlier lineages' archived records and experiment texts were written before DESIGN.md
    # section 11: their sentences are cleaned for the page (plain_numbers.plain_tree)
    from .plain_numbers import plain_tree
    # developmental_status names the September 13 definitions page itself (it named the 2026-09-17
    # tree until the dashboard staleness check of 2026-09-24)
    development=developmental_status(root,work,records)
    # held comparisons (manifest 'held') show their reason, never an out-of-reach due step
    checks=with_holds(root,read_json_or(root/ABILITY_STATUS,{}))
    results=guarded('the scheduled comparison results', lambda: ability_results(root,checks), lambda why: dict(rows=[],error=why))
    catalog=guarded('the capability programme', lambda: goal_catalog(root,checks,now,results), lambda why: None)
    brief=guarded('the work timeline', lambda: work_timeline(root,training,record,args,now,verdict['state']=='running'),
                  lambda why: dict(experience='Training timing is unavailable: '+why, events=[]))
    brief=guarded('the comparison schedule', lambda: comparison_schedule(brief,results,(catalog or {}).get('activity'),record,now,
                                                                         verdict['state']=='running'), lambda why: brief)
    panel=guarded('the scheduled results on the upgrade cards', lambda: link_scheduled(panel,results), lambda why: panel)
    return dict(schema=1,generated_at=now,training=training,abilities=plain_tree(abilities),
                automatic_ability_checks=checks,ability_results=results,
                goal_catalog=catalog,
                verdict=verdict,headline=headline,what_changed=changed,what_next=next_step,
                now_summary=now_summary(record,args,panel),
                training_brief=brief,
                project_records=guarded('the project records', lambda: project_records(root,now), lambda why: []),
                explainer=guarded('the explainer build time', lambda: explainer_state(root,training), lambda why: None),
                gain_of_function=panel,liveness=liveness,observations=observations,
                # the Over time tab's earlier-lineage texts carry 24-hour times written elsewhere
                work=am_pm_tree(plain_tree(work)),development=am_pm_tree(plain_tree(development)),
                legacy=legacy_state(root,now),
                records=plain_tree(records),
                videos=guarded('the recordings list', lambda: videos_for(root,bounds,training.get('run')), lambda why: []),
                live=live,
                evidence_registry=file_url(root,registry_path or root/'docs/development_dashboard/evidence.json'),
                legacy_dashboard='/files/Saved/mimo_bridge/dashboard_legacy.html',
                legacy_lifetime='/files/Saved/mimo_bridge/lifetime.html')


def lineage_payload(root=REPO, all_series=False, now=None):
    """GET /api/lineage: the lineage's gauges (tools/lineage_series.py), read incrementally."""
    from . import lineage_series
    keys = "all" if all_series else None
    try:
        return lineage_series.build(root, keys=keys, now=now, cache_dir=LINEAGE_CACHE_DIR)
    except (OSError, ValueError, KeyError, TypeError):
        raise
    except Exception as exc:   # noqa: BLE001 - becomes the route's 503 with the reason logged
        raise ValueError(f'{type(exc).__name__}: {exc}') from exc


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.new')
    tmp.write_text(text,encoding='utf-8')
    os.replace(tmp,path)


def install_entry(root, port):
    saved = root/'Saved/mimo_bridge'
    target, legacy = saved/'dashboard.html',saved/'dashboard_legacy.html'
    if target.exists() and 'development-dashboard-launcher' not in target.read_text(encoding='utf-8'):
        # Never overwrite an earlier preserved copy during later restarts.
        if not legacy.exists():
            shutil.copy2(target,legacy)
    url = f'http://127.0.0.1:{port}/'
    atomic_write(target, '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<!-- development-dashboard-launcher -->'
        f'<meta http-equiv="refresh" content="0;url={url}">'
        '<title>UEMA development</title><body>'
        f'<p><a href="{url}">Open the live development dashboard</a></p>'
        '<p><a href="dashboard_legacy.html">Old dashboard</a> · '
        '<a href="lifetime.html">Old whole-life page</a></p></body></html>')


def range_bounds(header, size):
    if not header:
        return 0,size-1,False
    match = re.fullmatch(r'bytes=(\d*)-(\d*)',header)
    if not match or not any(match.groups()):
        raise ValueError('Invalid byte range')
    start,end = match.groups()
    if not start:
        count=int(end)
        if count <= 0:
            raise ValueError('Empty suffix')
        start,end=max(0,size-count),size-1
    else:
        start,end=int(start),min(int(end),size-1) if end else size-1
    if start >= size or start > end:
        raise ValueError('Unsatisfiable range')
    return start,end,True


# THE ROUTES OF OCT 5, 2026 (the recovery directive's sections 5, 7, 9, 10, 12, 13) are tools/recovery_api.py's: the
# test registry audit, the component manifest's status, the defect register, the machine mode, the scoreboard, the
# canonical names and the dashboard's top in one read
from .recovery_api import ROUTES as RECOVERY_ROUTES  # noqa: E402

# THE MOVEMENT VIEWER (the operator's request of Oct 18, 2026; tools/movement_viewer/api.py): requested versus actual
# movement for every saved command, over dates. Read-only on the recordings; a drawing request goes to the viewer service.
MOVEMENT_ROUTES = ('/api/movement/sources', '/api/movement/series', '/api/movement/command', '/api/movement/render',
                   '/api/movement/auto', '/api/movement/probes', '/api/movement/probe_trials')
API_ROUTES = ('/api/progress', '/api/lineage', '/api/live', '/api/observer', '/api/observer/history',
              '/api/tests') + RECOVERY_ROUTES + MOVEMENT_ROUTES
# WHERE THE LINEAGE GAUGES' CACHE FILES GO: None is the served checkout's own Saved/mimo_bridge/lineage_series. A second
# dashboard that reads production's data from another checkout (main: --root) names its own folder (--cache-dir), so it
# writes nothing under production's Saved.
LINEAGE_CACHE_DIR = None
# THE EXPLAINER PAGE A SECOND DASHBOARD SERVES (main: --explainer): None serves the checkout's own
# Saved/mimo_bridge/explainer.html. A second dashboard that shows explainer code not yet merged names its own build
# (built outside production's checkout); it is served at the same address, /files/Saved/mimo_bridge/explainer.html, so
# every link to the Architecture tab works unchanged, and only on that second dashboard.
EXPLAINER_PAGE = None


class Handler(BaseHTTPRequestHandler):
    server_version = 'UEMADevelopment/1'

    def log_message(self, format, *args):
        if args and str(args[1] if len(args)>1 else '').startswith('5'):
            super().log_message(format,*args)

    def do_HEAD(self):
        self.do_GET(head=True)

    def do_POST(self):
        host = self.headers.get('Host','').split(':')[0]
        if host not in ('127.0.0.1','localhost'):
            self.send_error(403)
            return
        route = unquote(urlsplit(self.path).path)
        if route not in ('/api/live', '/api/movement/render'):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length') or 0)
            if route == '/api/movement/render' and length > 10_000:
                # a drawing request is a few ids; /api/live keeps its own rules unchanged
                raise ValueError('Request too large')
            body = json.loads(self.rfile.read(length).decode('utf-8') or '{}')
            if route == '/api/movement/render':
                # a drawing of one saved command (recording id, body, model step): queued for the viewer service
                from .movement_viewer.api import render_request
                answer = render_request(body)
            else:
                training = training_state(self.server.root, time.time())
                answer = live_command(self.server.root, training, body)
            content = json.dumps(answer).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            message = json.dumps({"error": str(exc)}).encode('utf-8')
            self.send_response(400)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(message)))
            self.end_headers()
            self.wfile.write(message)

    def do_GET(self, head=False):
        # Bind to loopback and reject foreign hostnames (including DNS rebinding).
        host = self.headers.get('Host','').split(':')[0]
        if host not in ('127.0.0.1','localhost'):
            self.send_error(403)
            return
        path = unquote(urlsplit(self.path).path)
        try:
            if path in API_ROUTES:
                if path == '/api/progress':
                    state = build_state(self.server.root)
                elif path == '/api/tests':
                    # the Tests page: every row of docs/gain_of_function_v4/test_registry.json, read from the file on
                    # this request (tools/test_registry.py; the style guide "NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE")
                    # ?view=sections puts each row inside its section (QUEUED first) and leaves out "groups"
                    from .test_registry import dashboard_view
                    query = parse_qs(urlsplit(self.path).query or '')
                    state = dashboard_view(self.server.root, code_root=getattr(self.server, 'code_root', None),
                                           rows_in='sections' if (query.get('view') or [''])[0] == 'sections'
                                           else 'groups')
                elif path in RECOVERY_ROUTES:
                    from .recovery_api import respond
                    state = respond(path, parse_qs(urlsplit(self.path).query or ''), self.server.root,
                                    getattr(self.server, 'code_root', None))
                elif path == '/api/observer':
                    # the developmental observer's Development view (addendum v4 section 32);
                    # every sentence is built by tools/observer/view.py. ?at= shows a dated copy and
                    # ?compare= another copy (or now) beside it (observer/history.py)
                    from .observer.view import build as observer_view
                    query = parse_qs(urlsplit(self.path).query or '')
                    state = observer_view(str(self.server.root), at=(query.get('at') or [None])[0],
                                          compare=(query.get('compare') or [None])[0])
                elif path == '/api/observer/history':
                    from .observer.view import history_view
                    state = history_view(str(self.server.root))
                elif path == '/api/live':
                    training = training_state(self.server.root, time.time())
                    state = live_state(self.server.root, training, training.get('_record'), training.get('_args'))
                elif path in MOVEMENT_ROUTES:
                    from .movement_viewer import api as movement
                    from .movement_viewer.commands import Refused
                    query = parse_qs(urlsplit(self.path).query or '')
                    try:
                        state = {'/api/movement/sources': movement.sources_view,
                                 '/api/movement/series': movement.series_view,
                                 '/api/movement/command': movement.command_view,
                                 '/api/movement/render': movement.render_status,
                                 '/api/movement/auto': movement.auto_view,
                                 '/api/movement/probes': movement.probes_view,
                                 '/api/movement/probe_trials': movement.probe_trials_view}[path](query)
                    except (ValueError, Refused) as exc:
                        # a refused request says why (an unknown id, a model step where no command starts)
                        message = json.dumps({"error": str(exc)}).encode('utf-8')
                        self.send_response(400)
                        self.send_header('Content-Type','application/json; charset=utf-8')
                        self.send_header('Content-Length',str(len(message)))
                        self.end_headers()
                        if not head:
                            self.wfile.write(message)
                        return
                else:
                    state = lineage_payload(self.server.root, all_series='all=1' in (urlsplit(self.path).query or ''))
                content = json.dumps(state,ensure_ascii=False,allow_nan=False).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type','application/json; charset=utf-8')
                self.send_header('Content-Length',str(len(content)))
                self.send_header('Cache-Control','no-store')
                self.end_headers()
                if not head:
                    self.wfile.write(content)
                return
            if path == '/':
                file = ASSETS/'development_dashboard.html'
            elif path == '/api/movement/image':
                # one picture of the movement viewer: its drawing key and a picture name, both checked by pattern
                from .movement_viewer.api import image_path
                file = image_path(parse_qs(urlsplit(self.path).query or ''))
            elif path == '/api/movement/export':
                # one drawing as a single HTML file with its dates and sources (its key checked by pattern)
                from .movement_viewer.api import export_path
                file = export_path(parse_qs(urlsplit(self.path).query or ''))
            elif path.startswith('/assets/'):
                name = path.removeprefix('/assets/')
                # elimination_diagram.js and .css: the process-of-elimination diagram, shared with the explainer;
                # movement_viewer.js and .css: the movement viewer's panel on the Now tab
                if name not in ('development_dashboard.css','development_dashboard.js','development_dashboard_terms.js',
                                'elimination_diagram.js','elimination_diagram.css','movement_viewer.js',
                                'movement_viewer.css'):
                    raise ValueError('Unknown asset')
                file = ASSETS/name
            elif path.startswith('/files/') and EXPLAINER_PAGE and path == '/files/Saved/mimo_bridge/explainer.html':
                file = Path(EXPLAINER_PAGE)
            elif path.startswith('/files/'):
                file = confined(self.server.root,path.removeprefix('/files/'))
                # .jsonl only under docs (the component QA log): the runs' stats files under Saved are not served
                if file.suffix.lower() not in ('.html','.json','.md','.mp4','.jpg','.png','.gif') and not (
                        file.suffix.lower()=='.jsonl' and file.is_relative_to(self.server.root/'docs')):
                    raise ValueError('Not a public dashboard asset')
                allowed = (self.server.root/'docs',self.server.root/'Saved/mimo_bridge')
                if not any(file.is_relative_to(p) for p in allowed):
                    raise ValueError('Not a dashboard source')
                # a second dashboard (serve_beside): a docs file that only this code's own checkout holds (the
                # component manifest, the QA log, the defect register) is served from there
                code_root = getattr(self.server,'code_root',None)
                if code_root is not None and not file.is_file() and file.is_relative_to(self.server.root/'docs'):
                    own = confined(Path(code_root),path.removeprefix('/files/'))
                    if own.is_file() and own.is_relative_to(Path(code_root)/'docs'):
                        file = own
            elif path.startswith('/experiments/'):
                from .experiment_board import artifact
                parts=path.split('/')
                if len(parts)!=4:raise ValueError('Invalid experiment record route')
                file=artifact(self.server.root,parts[2],parts[3])
            else:
                self.send_error(404)
                return
            self.send_file(file,head)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if path in API_ROUTES:
                print(f'Dashboard data unavailable ({path}): {exc}',flush=True)
                self.send_error(503,'Current dashboard data unavailable')
            else:
                self.send_error(404)

    def send_file(self,path,head):
        if not path.is_file():
            self.send_error(404)
            return
        size=path.stat().st_size
        try:
            start,end,partial=range_bounds(self.headers.get('Range'),size)
        except ValueError:
            self.send_response(416)
            self.send_header('Content-Range',f'bytes */{size}')
            self.send_header('Content-Length','0')
            self.end_headers()
            return
        content_type=mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
        if path.suffix=='.md':
            content_type='text/plain; charset=utf-8'
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(max(0,end-start+1)))
        self.send_header('Cache-Control','no-store')
        self.send_header('Accept-Ranges','bytes')
        self.send_header('X-Content-Type-Options','nosniff')
        if partial:
            self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head:
            return
        with path.open('rb') as stream:
            stream.seek(start)
            remaining=end-start+1
            while remaining>0:
                chunk=stream.read(min(256_000,remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining-=len(chunk)


def serve_beside(root, port, cache_dir, write_log=None, explainer=None):
    """A SECOND DASHBOARD THAT READS ANOTHER CHECKOUT'S DATA AND WRITES NOTHING THERE (Oct 5, 2026). `root` is the
    checkout whose test registry, Saved/ and runs are served (production's); the code, the component manifest, the
    defect register and the QA log are this module's own checkout. Unlike main()'s own server it does not replace
    the bookmarked entry page (Saved/mimo_bridge/dashboard.html), does not write development_dashboard_service.json,
    does not start the updater of the two older pages, and keeps the lineage gauges' cache files in `cache_dir`.
    With `write_log` the process records every file it writes inside `root` there (write_watch.watch_writes): an
    empty log is the evidence that it wrote nothing into that checkout. With `explainer` (a built explainer page
    outside `root`) that page is served at /files/Saved/mimo_bridge/explainer.html in place of root's own."""
    global LINEAGE_CACHE_DIR, EXPLAINER_PAGE
    if explainer:
        EXPLAINER_PAGE = Path(explainer).resolve()
    if write_log:
        from mimo_bridge.tools import write_watch
        write_watch.watch_writes(root, write_log)
    LINEAGE_CACHE_DIR=Path(cache_dir)
    LINEAGE_CACHE_DIR.mkdir(parents=True,exist_ok=True)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.root=Path(root).resolve()
    server.code_root=REPO
    print(f'Development dashboard (code {REPO}, data {server.root}, cache {LINEAGE_CACHE_DIR}'
          + (f', explainer {EXPLAINER_PAGE}' if EXPLAINER_PAGE else '') + f'): http://127.0.0.1:{port}/',flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8766)
    parser.add_argument('--root',type=Path,default=None,
                        help='serve another checkout\'s data (its test registry, Saved/ and runs) with this checkout\'s '
                             'code, writing nothing there: see serve_beside')
    parser.add_argument('--cache-dir',type=Path,default=None,
                        help='with --root: where the lineage gauges\' cache files go (default: a folder under the '
                             'temporary folder)')
    parser.add_argument('--write-log',type=Path,default=None,
                        help='with --root: record every file this process writes inside that checkout in this file '
                             '(write_watch); an empty file is the evidence that it wrote nothing there')
    parser.add_argument('--explainer',type=Path,default=None,
                        help='with --root: serve this built explainer page at /files/Saved/mimo_bridge/explainer.html '
                             'in place of that checkout\'s own (a build of explainer code that is not merged yet)')
    args=parser.parse_args()
    if args.root is not None and args.root.resolve()!=REPO.resolve():
        import tempfile
        serve_beside(args.root,args.port,args.cache_dir or Path(tempfile.gettempdir())/'uema_dashboard_beside',
                     write_log=args.write_log,explainer=args.explainer)
        return
    if args.explainer is not None:
        parser.error('--explainer is for a second dashboard started with --root')
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.root=REPO
    # Qualify the data before replacing the bookmarked entry page.
    build_state()
    install_entry(REPO,args.port)
    atomic_write(SAVED/'development_dashboard_service.json',json.dumps(dict(
        pid=os.getpid(),port=args.port,url=f'http://127.0.0.1:{args.port}/',
        started_at=time.time(),module='mimo_bridge.tools.development_dashboard'),indent=2))
    print(f'Development dashboard: http://127.0.0.1:{args.port}/',flush=True)
    # the movement viewer's service (tools/movement_viewer/service.py): draws on request, keeps its index current
    try:
        from .movement_viewer.api import ensure_service
        ensure_service()
    except Exception as exc:
        print(f'Movement viewer service not started: {exc}',flush=True)
    stop=threading.Event()
    updater=threading.Thread(target=legacy_loop,args=(stop,),daemon=True,name='legacy-dashboard-updater')
    updater.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()


if __name__=='__main__':
    main()
