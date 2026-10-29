# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Run declared physical ability comparisons when their saved-checkpoint step is due.

This worker owns no training process. It runs one frozen experiment at a time,
keeps durable success/failure receipts, and retries an interrupted comparison.

TEST CLASSES (the operator's decision of 2026-09-25 9:09 AM PDT; measurements and the arithmetic of
the expected training break: docs/gain_of_function_v4/agents/scheduler_classes_2026-09-25.md). With
the trainer up no test fits beside it in memory, so every round costs a saved training pause. A
'frequent' task runs every every_steps updates and may pause training for its own window. A
'triggered' task runs when one of its declared triggers fires (a committed change to its declared
code, a change of the production flags, a scheduled task's result crossing its declared line, a live
gauge moving past its declared tolerance), joining the next saved window, or at its longest interval
(every_steps); it pauses training by itself only at that interval or once ride_along_steps have passed
since a trigger fired without a window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

from .training_continuation import OwnerLock, atomic_json, complete_candidates, pause_reason


def read(path, default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return {} if default is None else default


TASK_CLASSES=('frequent','triggered')
# what this scheduler can do that an older one could not; a manifest entry can wait for one (first_step)
HOLD_FEATURE='hold_training_on_failed_comparison'
# production flags that name a run's own files, not how it learns (the supervisor rewrites them on every
# relaunch: training_continuation.supervise)
PER_RUN_FLAGS=('resume','out','checkpoint_storage','caregiver_record_dir')
POLL_S=30
# while a saved window is open training stands still: the scheduler looks every 5 s instead of every
# 30 s (measured 2026-09-24/25: each test in a window waited 30 to 60 s for the next poll)
WINDOW_POLL_S=5
ACTIVE_STATES=('starting','running','process identity unavailable')
# the live gauges a triggered task may declare, each read from one 'mimo' record of the trainer's
# statistics and averaged over the newest GAUGE_RECORDS records of the active run (100 updates apart)
GAUGE_RECORDS=10


def first_step(task):
    """The task's first scheduled step. A task that holds training when its comparison fails
    (HOLD_FEATURE) keeps a first_step out of reach of a scheduler without that hold (every scheduler
    before 2026-09-25 reads the manifest each cycle) and names its real first step in
    first_step_under_hold."""
    if task.get(HOLD_FEATURE) is True and task.get('first_step_under_hold') is not None:
        return int(task['first_step_under_hold'])
    return int(task['first_step'])


def task_class(task):
    return task.get('class','frequent')


def due_checkpoint(task, record, candidates, threshold=None):
    """Advance only after a completed result; never skip a missed milestone. `threshold` is the first
    step the task may run at this cycle (plan_tasks); without it, its regular due step."""
    if threshold is None:
        threshold=max(first_step(task),int(record.get('completed_step',-int(task['every_steps'])))+int(task['every_steps']))
    for step,path in sorted(candidates):
        if step>=threshold:return step,path
    return None


def trainer_running():
    """Whether a production trainer process exists (training_continuation's own test)."""
    from .training_continuation import processes
    return bool(processes('trainer'))


def latest_step(run):
    path=run/'replay_training_stats.jsonl'
    if not path.is_file():return None
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-1024*1024))
        lines=stream.read().decode('utf-8',errors='replace').splitlines()
    for line in reversed(lines):
        try:record=json.loads(line)
        except ValueError:continue
        if record.get('type')=='mimo':return record.get('step')
    return None


_RECORDS={}


def latest_records(run, count=GAUGE_RECORDS):
    """The newest `count` 'mimo' records of a run's statistics, oldest first; read again only when the
    file changes. A record is about 200 KB (2026-09-25), so the tail read is sized to hold them."""
    path=Path(run)/'replay_training_stats.jsonl'
    try:
        stat=path.stat()
    except OSError:
        return []
    key=(str(path),count)
    cached=_RECORDS.get(key)
    if cached and cached[0]==(stat.st_size,stat.st_mtime_ns):
        return cached[1]
    found=[]
    try:
        with path.open('rb') as stream:
            stream.seek(max(0,stat.st_size-(count+2)*300_000))
            lines=stream.read().split(b'\n')
    except OSError:
        return []
    for line in reversed(lines):
        if b'"type": "mimo"' not in line[:40]:
            continue
        try:record=json.loads(line)
        except ValueError:continue
        found.append(record)
        if len(found)==count:break
    found.reverse()
    if len(_RECORDS)>16:_RECORDS.clear()
    _RECORDS[key]=((stat.st_size,stat.st_mtime_ns),found)
    return found


def confined(root, relative):
    result=(root/relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Ability test artifact must stay inside the workspace')
    return result


PATH_FLAG=re.compile(r'--[a-z][a-z0-9-]*')


def path_arguments(root, task, checkpoint=None):
    """A script that needs more than its checkpoint and output folder declares each extra input as a
    named repository file (manifest 'path_arguments', e.g. the remembered hand-goal comparison's
    --calibration). Each is confined like the script and passed as an absolute path, so a result
    can be bound to the file the scheduler passed. Raises ValueError for an entry that cannot run.
    With `checkpoint` (task_command), a flag CHECKPOINT_CALIBRATION_ARGUMENTS names for the task's script
    passes the hand calibration that checkpoint saved (checkpoint_calibration)."""
    extra=[]
    bound=CHECKPOINT_CALIBRATION_ARGUMENTS.get(Path(str(task.get('script') or '')).name,())
    for flag,relative in sorted((task.get('path_arguments') or {}).items()):
        if not PATH_FLAG.fullmatch(str(flag)) or flag in ('--checkpoint','--out'):
            raise ValueError(f'Ability test path argument {flag!r} is not a plain long option')
        path=confined(root,str(relative))
        if not path.is_file():
            raise ValueError(f'Ability test path argument {flag} names a missing file: {relative}')
        if checkpoint is not None and flag in bound:
            path=checkpoint_calibration(root,path,checkpoint)
        extra+=[flag,str(path)]
    return extra


# THE HAND CALIBRATION A CHECKPOINT SAVED (Oct 9, 2026; the matched comparison of the finger fix, the operator's decision
# received by 11:53 PM PDT, Oct 8). The remembered hand-goal comparison builds the checkpoint's learner with the
# --calibration it is given, and the trainer's load refuses a calibration other than the one the checkpoint saved in its
# model_receipt.json (hand_response) unless an admission record names it (LemLite._restore_hand_response_calibration,
# lem.model.hand_response.admission_check). A run launched with --replace-saved-hand-calibration (the finger arm) saves
# the replacing calibration, the refit, in every checkpoint it writes, so the manifest's file, the admitted fit, would be
# refused at load on each of them and every round would end without a result. For the flags named here the scheduler
# passes the file whose bytes the checkpoint's receipt names (hand_response.calibration_file_sha256): the manifest's file
# when it is that one, as on every checkpoint before a replacement (the command is unchanged); else the file the
# checkpoint's run launched with (its args.json hand_response_calibration, inside the workspace) when its bytes and its
# payload hash are the receipt's; else the manifest's file as before, and the load's refusal ends the attempt by the
# retry rules. A checkpoint without a readable receipt or a saved hand calibration gets the manifest's file.
CHECKPOINT_CALIBRATION_ARGUMENTS={'compare_hand_memory_route_2026-09-24.py':('--calibration',)}


def checkpoint_calibration(root, declared, checkpoint):
    """The hand calibration file to pass with `checkpoint` in place of `declared`, the manifest's file (the rule
    above). Never raises: what cannot be read leaves `declared`."""
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    checkpoint=Path(str(checkpoint))
    try:
        saved=json.loads((checkpoint/'model_receipt.json').read_text(encoding='utf-8')).get('hand_response')
        if not isinstance(saved,dict) or not isinstance(saved.get('calibration_file_sha256'),str):
            return declared
        if sha(declared)==saved['calibration_file_sha256']:
            return declared
        # the run that saved the checkpoint: its lineage.json names it (a checkpoint stored outside its run folder)
        out=read(checkpoint/'lineage.json').get('out')
        launched=read((Path(str(out)) if out else checkpoint.parent.parent)/'args.json').get('hand_response_calibration')
        if launched:
            candidate=confined(root,str(launched))
            if (candidate.is_file() and sha(candidate)==saved['calibration_file_sha256']
                    and json.loads(candidate.read_text(encoding='utf-8')).get('sha256')==saved.get('sha256')):
                return candidate
    except (OSError,ValueError,TypeError,AttributeError):
        pass
    return declared


PLAIN_VALUE=re.compile(r'[A-Za-z0-9_.+][A-Za-z0-9_.+-]*')


def plain_arguments(task):
    """A script's plain options (manifest 'arguments', a list, e.g. the body comparison's
    ["--goal-suite", "hand_and_body"]): each token is a plain long option or a plain value after one -
    letters, digits, '_', '.', '+' and inner '-', never a path, a '..' or an '=' - and no option is the
    checkpoint, the output folder, a path argument or given twice. A file input goes through
    path_arguments, which confines it to the workspace. Raises ValueError for an entry that cannot run."""
    arguments=task.get('arguments')
    if arguments is None:
        return []
    if not isinstance(arguments,list) or not all(isinstance(token,str) for token in arguments):
        raise ValueError('Ability test arguments must be a list of plain words')
    taken={'--checkpoint','--out'}|{str(flag) for flag in (task.get('path_arguments') or {})}
    seen=set()
    for position,token in enumerate(arguments):
        if token.startswith('-'):
            if not PATH_FLAG.fullmatch(token) or token in taken or token in seen:
                raise ValueError(f'Ability test argument {token!r} is not a plain long option of its own')
            seen.add(token)
        elif position==0 or not PLAIN_VALUE.fullmatch(token) or '..' in token:
            raise ValueError(f'Ability test argument {token!r} is not a plain value after an option')
    return list(arguments)


def dependency_paths(root, task):
    """The repository files a test declares it runs on besides its script (manifest 'depends': the
    probe harness, a probe's own helper module). Each is confined like the script and must exist.
    Raises ValueError for an entry that cannot run."""
    paths=[]
    for relative in sorted(str(r) for r in (task.get('depends') or [])):
        path=confined(root,relative)
        if not path.is_file():
            raise ValueError(f'Ability test dependency is missing: {relative}')
        paths.append((relative,path))
    return paths


def test_version(root, task, script_digest):
    """What repairing a failing test must change: its script and every file it declares in 'depends'.

    FAILURES COUNT PER VERSION, NOT PER CHECKPOINT (2026-09-24). Counted per checkpoint, the eye test
    failed twice at step 397,050 with the same code, and after the next resume (a new run folder, so a
    new checkpoint candidate) the count started again at 0: the scheduler stopped training at 9:12 PM
    PDT for two more attempts of code already known to fail. And a count keyed on the script alone was
    never reset by a fix in the probe harness, where the eye test's repair lives. Two failures in a row
    of one version now hold the test whatever the checkpoint, until the script or a declared file
    changes. The manifest's plain 'arguments' are part of the version too: a changed option is a
    different test (a task without them keeps the version it had before they existed)."""
    digest=hashlib.sha256(script_digest.encode('ascii'))
    for relative,path in dependency_paths(root,task):
        digest.update(relative.encode('utf-8'))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    arguments=plain_arguments(task)
    if arguments:
        digest.update(json.dumps(arguments).encode('utf-8'))
    return digest.hexdigest()


# A probe that refuses to run under a scheduler older than its validator (probe_planner_candidates
# stale_scheduler) says so in its log. An out-of-date scheduler still counts that refusal as a failed
# attempt - two of them hold the test with 'repair needed', which also releases the comparison window
# so training resumes (ability_comparison_window). But a hold keyed on the test version survived the
# restart that cures it: neither the script nor a declared file had changed (review of the ranking
# change, 2026-09-25). A scheduler that is current takes those refusals back and runs the test.
STALE_REFUSAL='started before the last change to tools/ability_rechecks.py'


def refused_as_stale(record):
    """Whether an attempt ended with its probe refusing its scheduler as older than its validator."""
    try:
        log=Path(record['output']).with_suffix('.log')
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-8192))
            return STALE_REFUSAL in stream.read().decode('utf-8',errors='replace')
    except (OSError,KeyError,TypeError,ValueError):
        return False


def scheduler_is_current(process=None, source=None):
    """Whether this scheduler started after the last change to its own code file: the probes' own
    test (probe_planner_candidates.stale_scheduler), from the scheduler's side."""
    try:
        started=(process or psutil.Process()).create_time()
        return started>=Path(source or __file__).stat().st_mtime
    except (psutil.Error,OSError):
        return False


def forgive_stale_refusals(record):
    """Take back the failures and attempts of the latest attempts that were stale-scheduler refusals
    (an out-of-date scheduler counted them). Call only from a current scheduler (scheduler_is_current):
    an out-of-date one would relaunch a probe that refuses it again. Returns how many were taken back."""
    history=[h for h in (record.get('history') or []) if isinstance(h,dict)]
    refused=0
    for receipt in reversed(history):
        if receipt.get('status')=='failed' and not receipt.get('stale_refusal_forgiven') and refused_as_stale(receipt):
            receipt['stale_refusal_forgiven']=True
            refused+=1
        else:
            break
    if refused:
        record['failures_in_a_row']=max(0,int(record.get('failures_in_a_row',0))-refused)
        if not record['failures_in_a_row']:record['failed_version']=None
        record['attempts']=max(0,int(record.get('attempts',0))-refused)
    return refused


# ------------------------------------------------ triggers (docs/gain_of_function_v4/agents/scheduler_classes_2026-09-25.md)
def git(root, *arguments):
    """git's standard output, or None when git cannot answer (never raises)."""
    try:
        done=subprocess.run(['git','-C',str(root),*arguments],capture_output=True,text=True,timeout=30,
                            stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    except (OSError,subprocess.SubprocessError,ValueError):
        return None
    return done.stdout if done.returncode==0 else None


def code_head(root):
    out=git(root,'rev-parse','HEAD')
    head=(out or '').strip()
    return head if re.fullmatch(r'[0-9a-f]{40}',head) else None


def commit_before(root, when):
    """The commit the production branch held at time `when` (epoch seconds), or None: HEAD's newest
    first-parent ancestor committed by then. Without --first-parent a merged branch's own commits count
    too, with their own times, and the answer can be a commit production never ran (review of 2026-09-25:
    for a round started at 3:30 AM it gave 50177c48e, merged only at 9:48 AM)."""
    out=git(root,'rev-list','--first-parent','-1','--before=@%d'%int(when),'HEAD')
    head=(out or '').strip()
    return head if re.fullmatch(r'[0-9a-f]{40}',head) else None


_CHANGED={}


def changed_code(root, basis, head, paths):
    """The declared files whose committed content differs between the basis commit and HEAD: a sorted
    list (empty when none), or None when either commit is unknown or git cannot tell."""
    if not basis or not head or not paths:
        return None
    if basis==head:
        return []
    key=(basis,head,tuple(paths))
    if key not in _CHANGED:
        out=git(root,'diff','--name-only',basis,head,'--',*paths)
        if out is None:
            return None
        if len(_CHANGED)>256:_CHANGED.clear()
        _CHANGED[key]=sorted({line.strip() for line in out.splitlines() if line.strip()})
    return _CHANGED[key]


def production_flags(run):
    """A digest of the run's flags (args.json) without the ones naming its own files, or None."""
    try:
        args=json.loads((Path(run)/'args.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return None
    if not isinstance(args,dict):
        return None
    kept={k:v for k,v in args.items() if k not in PER_RUN_FLAGS}
    return hashlib.sha256(json.dumps(kept,sort_keys=True,default=str).encode('utf-8')).hexdigest()


def _eye_forecast_share_150ms(record):
    e=record['efference']
    return 1.0-float(e['gaze_deg_err'][2])/float(e['gaze_deg_do_nothing'][2])


def _hand_joint_error_deg(record):
    return float(record['hand_control']['joint_error_deg'])


# name: (reader, meaning). Tolerances are declared in the manifest from the measured movement of the
# same average (the design record above): 10 records span 1,000 training updates.
GAUGES={
    'eye_forecast_share_150ms':(_eye_forecast_share_150ms,
        "share of the eye-angle error of 'the eyes stay still' that the live eye forecast removes 150 ms "
        "ahead (0 = none, 1 = all), from the trainer's efference record"),
    'hand_joint_error_deg':(_hand_joint_error_deg,
        "production hands' mean joint error from their joint goals, degrees, from the trainer's hand_control record"),
}


def gauge_values(run):
    """Each gauge's mean over the active run's newest GAUGE_RECORDS records; a gauge with fewer
    readable records (a run younger than about 1,000 updates) is left out."""
    records=latest_records(run,GAUGE_RECORDS)
    out={}
    for name,(read_one,_) in GAUGES.items():
        values=[]
        for record in records:
            try:
                value=read_one(record)
            except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError):
                continue
            if math.isfinite(value):
                values.append(value)
        if len(values)>=GAUGE_RECORDS:
            out[name]=dict(value=sum(values)/len(values),records=len(values),step=records[-1].get('step'))
    return out


def declared_line(script, payload):
    """Whether a validated result meets its comparison's own declared line, as the dashboard reads the
    same result (development_dashboard._won): True, False, or None for a script without one."""
    if script=='qualify_hand_execution_2026-09-24.py':
        s=payload['summary']
        return all(s[g]['moving_wrist']<s[g]['fixed_world'] for g in payload['goals'])
    if script=='compare_hand_body_goals_2026-09-24.py':
        against=payload['summary']['physical_goal_error']['against_controller']
        return any(against[k]['paired_bootstrap95'][1]<0 for k in ('synergy','direct'))
    # the fixed-start probe (fixed_probe_set_2026-10-18.py, task fixed_start_body_and_synergy): THE GAIN RULE (proxy
    # decision movement-viewer-fixed-probe-decision-20261018 under the operator's October 13, 2026 delegation; before it, the
    # body comparison's line above, which compared the anchored planner searches with each checkpoint's own controller
    # command). The line is the direct arm's against the same arm of the bound reference run: met when its 95 percent
    # interval lies below 0. A valid comparison that does not meet it reads False; a comparison that is not valid has no line
    # (None, which registry_finished writes INVALID; validate_result refuses such a result before this is read)
    if script=='fixed_probe_set_2026-10-18.py':
        gain=payload.get('gain')
        if not isinstance(gain,dict) or gain.get('rule_version')!=FIXED_START_GAIN_RULE or gain.get('status')!='valid':
            return None
        line=gain.get('line_met')
        return line if isinstance(line,bool) else None
    if script=='probe_saved_gaze.py':
        eye=payload['summary']['saved_starts']['150ms']['eye_prediction_error_degrees']
        return eye.get('candidate_error_lower_beyond_sample_uncertainty') is True
    if script=='probe_saved_self_touch_choices.py':
        return (payload.get('passed') if payload.get('probe') is not None
                else payload.get('passed_private_comparison')) is True
    if script in ('probe_planner_candidates.py','compare_hand_memory_route_2026-09-24.py'):
        return payload.get('passed') is True
    return None


def line_of(record):
    """The declared line of a completed attempt's result file, or None when it cannot be read."""
    try:
        payload=read(record['result'])
        return declared_line(Path(record['command'][2]).name,payload)
    except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError):
        return None


def validate_schedule(root, task, ids, ride_along_steps):
    """A task's class, cadence, triggers and hold declaration; raises ValueError for an entry that
    cannot be scheduled (it is then held with the reason, like an entry that cannot launch)."""
    if task_class(task) not in TASK_CLASSES:
        raise ValueError(f"Ability test class {task.get('class')!r} is not one of {TASK_CLASSES}")
    for key in ('first_step','every_steps'):
        if not isinstance(task.get(key),int) or isinstance(task.get(key),bool) or task[key]<(1 if key=='every_steps' else 0):
            raise ValueError(f'Ability test {key} must be a whole number of training updates')
    need=task.get('memory_gib')
    if not isinstance(need,(int,float)) or isinstance(need,bool) or not math.isfinite(need) or need<=0:
        raise ValueError('Ability test memory_gib must be a positive number of GiB')
    if task.get(HOLD_FEATURE) not in (None,True,False):
        raise ValueError(f'Ability test {HOLD_FEATURE} must be true or false')
    under=task.get('first_step_under_hold')
    if under is not None and (not isinstance(under,int) or isinstance(under,bool) or under<0 or task.get(HOLD_FEATURE) is not True):
        raise ValueError('first_step_under_hold needs a whole step and hold_training_on_failed_comparison true')
    if not isinstance(ride_along_steps,int) or isinstance(ride_along_steps,bool) or ride_along_steps<1:
        raise ValueError('ride_along_steps must be a positive whole number of training updates')
    triggers=task.get('triggers')
    if triggers is None:
        return
    if task_class(task)!='triggered' or not isinstance(triggers,dict):
        raise ValueError('Only a triggered ability test declares triggers, as an object')
    unknown=set(triggers)-{'code','flags','gauges','lines'}
    if unknown:
        raise ValueError(f'Unknown ability test triggers: {sorted(unknown)}')
    code=triggers.get('code') or []
    if not isinstance(code,list) or not all(isinstance(p,str) and p and '\\' not in p for p in code):
        raise ValueError('Code triggers must be a list of repository paths written with /')
    for relative in code:
        if not confined(root,relative).exists():
            raise ValueError(f'Code trigger names a missing path: {relative}')
    if triggers.get('flags') not in (None,True,False):
        raise ValueError('The flags trigger must be true or false')
    for gauge in triggers.get('gauges') or []:
        if not (isinstance(gauge,dict) and gauge.get('name') in GAUGES and isinstance(gauge.get('tolerance'),(int,float))
                and not isinstance(gauge['tolerance'],bool) and math.isfinite(gauge['tolerance']) and gauge['tolerance']>0):
            raise ValueError(f'A gauge trigger needs a known name ({sorted(GAUGES)}) and a positive tolerance')
    lines=triggers.get('lines') or []
    if not isinstance(lines,list) or not all(isinstance(t,str) and t in ids for t in lines):
        raise ValueError('Line triggers must name declared ability tests')


def trigger_context(root, run, manifest):
    """What this cycle's triggers compare against: HEAD, the active run's flags, the live gauges and
    step. Never raises; a reading that fails is None and fires nothing."""
    context=dict(root=root,head=None,flags=None,gauges={},live_step=None,
                 ride_along_steps=manifest.get('ride_along_steps'))
    declared=[t.get('triggers') or {} for t in manifest.get('tasks',[])
              if isinstance(t,dict) and task_class(t)=='triggered' and isinstance(t.get('triggers') or {},dict)]
    # only the readings some task declares (git is a process per cycle)
    readings=[('live_step',lambda:latest_step(Path(run)))]
    if any(t.get('code') for t in declared):
        readings.append(('head',lambda:code_head(root)))
    if any(t.get('flags') for t in declared):
        readings.append(('flags',lambda:production_flags(run)))
    if any(t.get('gauges') for t in declared):
        readings.append(('gauges',lambda:gauge_values(run)))
    for key,read_it in readings:
        try:
            context[key]=read_it()
        except Exception:                # a trigger reading never costs the scheduler its cycle
            pass
    if not isinstance(context['gauges'],dict):
        context['gauges']={}
    return context


def backfill(root, task, record, context):
    """A task that completed rounds under a scheduler without triggers gets the basis of its last
    completed round: the commit that was HEAD when it started, the flags of the run whose checkpoint it
    tested, and the live gauges as they read now (the older values are not recorded); and its
    declared-line history from its two newest completed results."""
    triggers=task.get('triggers') or {}
    history=[h for h in (record.get('history') or []) if isinstance(h,dict) and h.get('status')=='complete']
    history.sort(key=lambda h:(h.get('attempt_step') or 0,h.get('completed_at') or 0))
    if 'basis' not in record and history:
        last=history[-1]
        run=Path(str(last.get('checkpoint') or '')).parent.parent
        record['basis']=dict(code_commit=commit_before(root,last.get('started_at') or 0) if triggers.get('code') else None,
                             flags=production_flags(run) if triggers.get('flags') else None,
                             gauges={k:v['value'] for k,v in context['gauges'].items()},
                             backfilled=True)
    if 'line_met' not in record and history:
        lines=[line_of(h) for h in history[-2:]]
        record['line_met']=lines[-1]
        if len(lines)==2 and isinstance(lines[0],bool) and isinstance(lines[1],bool) and lines[0]!=lines[1]:
            record['line_changed']=dict(step=history[-1].get('attempt_step'),at=history[-1].get('completed_at'),
                                        before=lines[0],after=lines[1])


def trigger_reasons(task, record, context, tasks):
    """{kind: plain reason} for every trigger of a triggered task that fires now."""
    triggers=task.get('triggers') or {}
    basis=record.get('basis') if isinstance(record.get('basis'),dict) else {}
    reasons={}
    changed=changed_code(context['root'],basis.get('code_commit'),context.get('head'),triggers.get('code') or [])
    if changed:
        reasons['code']=('committed code changed since its last round: '+', '.join(changed[:4])
                         +(f' and {len(changed)-4} more' if len(changed)>4 else ''))
    if triggers.get('flags') and basis.get('flags') and context.get('flags') and basis['flags']!=context['flags']:
        reasons['flags']="the production run's flags differ from those of the run its last round tested"
    for gauge in triggers.get('gauges') or []:
        now=context['gauges'].get(gauge['name'])
        then=(basis.get('gauges') or {}).get(gauge['name'])
        if now and isinstance(then,(int,float)) and abs(now['value']-then)>float(gauge['tolerance']):
            reasons['gauge '+gauge['name']]=(f"live gauge {gauge['name']} moved from {then:.3f} at its last round to "
                                             f"{now['value']:.3f} (declared tolerance {gauge['tolerance']:g})")
    completed=int(record['completed_step'])
    for other in triggers.get('lines') or []:
        change=(tasks.get(other) or {}).get('line_changed')
        step=change.get('step') if isinstance(change,dict) else None
        if isinstance(step,int) and (step>completed or (other==task['id'] and step==completed)):
            title=(tasks.get(other) or {}).get('title') or other
            reasons['line '+other]=(f"{title} went from {'meeting' if change.get('before') else 'missing'} its declared "
                                    f"line to {'meeting' if change.get('after') else 'missing'} it at step {step:,}")
    return reasons


def fired_triggers(task, record, context, tasks):
    """The plain reasons a triggered task is due before its longest interval (empty when none)."""
    return list(trigger_reasons(task,record,context,tasks).values())


def schedule(task, record, context, tasks):
    """(run, open, reasons): the first checkpoint step the task may run at, the first step at which it
    may pause training itself, and why a triggered task runs early.

    A FIRED TRIGGER HOLDS UNTIL THE TASK'S NEXT COMPLETED ROUND (review of 2026-09-25). A window lists its
    tests when it is requested; a trigger that stopped firing before its test launched (a gauge average
    back inside its tolerance, a reverting commit) left the test unable to run on the window's checkpoint.
    THE RIDE-ALONG WAIT STARTS WHEN A LIVE STEP IS KNOWN (same review): a relaunched segment has no
    record for its first 100 updates, and a trigger first seen then (the flags trigger always is) started
    its wait at the last completed step, so the task opened its own pause as soon as the segment reported
    instead of joining the next one. Until a live step is known the task only joins pauses."""
    every=int(task['every_steps'])
    start=first_step(task)
    completed=record.get('completed_step')
    longest=max(start,(int(completed) if completed is not None else -every)+every)
    if task_class(task)!='triggered' or completed is None:
        record.pop('triggered',None)
        return longest,longest,[]
    fired=record.get('triggered') if isinstance(record.get('triggered'),dict) else {}
    kinds=dict(fired.get('kinds') or {}) if isinstance(fired.get('kinds'),dict) else {}
    for kind,text in trigger_reasons(task,record,context,tasks).items():
        kinds.setdefault(kind,text)
    if not kinds:
        record.pop('triggered',None)
        return longest,longest,[]
    since=fired.get('since_step')
    if not isinstance(since,int) or isinstance(since,bool):
        since=context['live_step'] if isinstance(context.get('live_step'),int) else None
    reasons=list(kinds.values())
    record['triggered']=dict(since_step=since,since=fired.get('since') or time.time(),kinds=kinds,reasons=reasons)
    opened=longest if since is None else min(longest,max(start,since+int(context['ride_along_steps'])))
    return max(start,int(completed)+1),opened,reasons


def launch_gauges(record, context):
    """The live gauge values a round's basis keeps: this cycle's readings, and for a gauge the active
    segment is too young to read (fewer than GAUGE_RECORDS records) the previous basis value. A round on a
    young segment left an empty basis before 2026-09-25, and its gauge trigger could not fire until the
    round after (review of that day)."""
    basis=record.get('basis') if isinstance(record.get('basis'),dict) else {}
    kept={k:v for k,v in (basis.get('gauges') or {}).items() if isinstance(v,(int,float)) and not isinstance(v,bool)}
    kept.update({k:v['value'] for k,v in (context.get('gauges') or {}).items()})
    return kept


# NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE (the style guide, operator 2026-10-02): a task runs only while
# docs/gain_of_function_v4/test_registry.json holds one complete row whose id is the task's id (tools/test_registry.py
# judges it). The file is read every cycle, like the manifest: a task without a complete row is held as 'repair
# needed' with the reason, like an entry that cannot launch (plan_tasks), so it never requests a checkpoint or a
# window; it runs again on the first cycle after its row is complete.
REGISTRY_REASON=('The test registry refuses this task (docs/gain_of_function_v4/test_registry.json): {why}. The '
                 'scheduler runs this task once its row is complete and neither a block in its row nor the defect '
                 "register's 24-hour rule holds it (the style guide: NO TEST RUNS WITHOUT ITS ROW ON THE TESTS PAGE; the "
                 'dashboard review of Oct 5, 2026, finding F3).')


def registry_refusals(root, task_ids):
    """{task id: why it may not run} for every task the test registry refuses (test_registry.refusal: no complete row,
    a block written into its row, or the defect register's 24-hour rule), from one read of the file now; {} when it
    refuses none."""
    from .test_registry import refusals, registry_path
    return refusals(task_ids,path=registry_path(root))


# THE SCHEDULER WRITES ITS ROUNDS INTO THEIR ROWS (Oct 5, 2026; the recovery directive's section 8: a registry
# generated by execution). Until then it only read the rows (registry_refusals) and a scheduled check ran while its
# row said queued. A round's start writes status running with the child's process id and creation time, the
# checkpoint and the run id whose mark the child's environment carries (run_task); its end writes PASS when the
# validated result met its declared line, FAIL when it did not, and INVALID when the round ended without a complete
# valid result or without a line that could be read. Both go through test_registry's runner functions, which never
# raise: a registry that cannot be written costs the round nothing, and test_runner.audit() then reports the mismatch.
REGISTRY_RUNNER='ability_rechecks.py'


def registry_started(root, task_id, record, child):
    """Write a launched round into its row; returns why the write failed, or None."""
    from .test_registry import job_started, registry_path
    destination=Path(record['output'])
    return job_started(task_id,started=record.get('started_at'),log=destination.with_suffix('.log'),
        runner=REGISTRY_RUNNER,receipt=destination.with_suffix('.receipt.json'),path=registry_path(root),
        pid=child.get('pid'),pid_created=child.get('create_time'),checkpoint=record.get('checkpoint'),
        run_id=record.get('registry_run_id'),command=record.get('command'),cwd=root,owner='the ability scheduler')


def registry_finished(root, task_id, record, outcome):
    """Write a finished round into its row; returns (why the write failed or None, the state written or None)."""
    from .test_registry import finish_job, registry_path
    if outcome.get('status')=='complete':
        line=outcome.get('line_met')
        state='PASS' if line is True else 'FAIL' if line is False else 'INVALID'
        basis=("the scheduler's validator accepted %s; its declared line was %s"
               %(outcome.get('result'),'met' if line is True else 'not met' if line is False else 'not readable'))
    else:
        state,basis='INVALID',"the round ended without a complete valid result: %s"%outcome.get('reason')
    return finish_job(task_id,started=record.get('started_at'),ended=record.get('completed_at'),
        log=Path(record['output']).with_suffix('.log'),runner=REGISTRY_RUNNER,
        receipt=Path(record['output']).with_suffix('.receipt.json'),path=registry_path(root),
        run_id=record.get('registry_run_id'),root=root,state=state,judged_by=basis,result_file=outcome.get('result'))


def open_review(record):
    """A failed round awaiting the operator's review (HOLD_FEATURE), or None."""
    review=record.get('review_hold')
    return review if isinstance(review,dict) and not review.get('released_at') else None


def plan_tasks(root, manifest, state, run, current):
    """Every declared task's schedule for this cycle, written into its status record before attempts
    are recovered or a window advances (both read these fields): due_step, the step from which the task
    may pause training itself; ride_step, the earlier step from which a triggered task joins a saved
    window; and its hold, with the status that says why. A held task never requests a checkpoint or a
    window: an entry that cannot launch, a test version that failed twice in a row, or a failed round
    awaiting review. STATUSES ARE RECOMPUTED EVERY CYCLE (2026-09-25): a 'repair needed' left after a
    current scheduler took back stale refusals kept the planner test marked held from 9:10 to 9:14 AM,
    and a window's pending list skips tasks marked held. Never raises."""
    context=trigger_context(root,run,manifest)
    ids=[str(t.get('id')) for t in manifest.get('tasks',[]) if isinstance(t,dict)]
    try:
        unregistered=registry_refusals(root,ids)
    except Exception as error:          # never raises: a registry that cannot be checked holds every task
        unregistered={i:f'the test registry could not be checked ({type(error).__name__}: {error})' for i in ids}
    plans,unnamed={},[]
    for task in manifest.get('tasks',[]):
        task_id=str(task.get('id')) if isinstance(task,dict) else ''
        if not re.fullmatch(r'[A-Za-z0-9_-]+',task_id):
            # the id names the task's output folders: an entry without a plain one never launches
            unnamed.append(task_id or repr(task)[:80])
            continue
        record=state['tasks'].setdefault(task_id,{})
        active=record.get('status') in ACTIVE_STATES
        try:
            record['title']=str(task['title'])
            # before any snapshot or memory window: an entry that cannot launch must neither pause
            # training for nothing nor stop the other declared comparisons
            plain_arguments(task)
            path_arguments(root,task)
            dependency_paths(root,task)
            validate_schedule(root,task,ids,manifest.get('ride_along_steps') if task_class(task)=='triggered' else 1)
            script=confined(root,task['script'])
            digest=hashlib.sha256(script.read_bytes()).hexdigest()
            version=test_version(root,task,digest)
        except (ValueError,OSError,KeyError,TypeError) as error:
            if not active:
                record.update(status='repair needed',reason=f'The manifest entry cannot be launched: {error}')
            plans[task_id]=dict(blocked=True)
            continue
        if unregistered.get(task_id):
            # a running round is never stopped for its row (it is reconciled as before); the next one is held
            if not active:
                record.update(status='repair needed',reason=REGISTRY_REASON.format(why=unregistered[task_id]))
            plans[task_id]=dict(blocked=True)
            continue
        try:
            # a probe's refusal of an out-of-date scheduler is not a failure of the test (refused_as_stale):
            # a current scheduler takes back what an out-of-date one counted
            if current:
                forgive_stale_refusals(record)
            if task_class(task)=='triggered':
                backfill(root,task,record,context)
            run_step,open_step,reasons=schedule(task,record,context,state['tasks'])
        except Exception as error:       # a trigger reading never costs the task its regular schedule
            record['trigger_error']=f'{type(error).__name__}: {error}'
            every=int(task['every_steps'])
            run_step=open_step=max(first_step(task),int(record.get('completed_step',-every))+every)
            reasons=[]
        record['due_step']=open_step
        if run_step<open_step:
            record['ride_step']=run_step
        else:
            record.pop('ride_step',None)
        held_version=record.get('failed_version')==version and int(record.get('failures_in_a_row',0))>=2
        review=open_review(record)
        if not active:
            if review:
                record.update(status='held for review',reason=review.get('reason'))
            elif held_version:
                record.update(status='repair needed',
                              reason='Two attempts of this version of the test failed in a row (the last at step %s); '
                                     'change the script, a file it declares in depends or its arguments and it retries '
                                     'automatically.' % record.get('attempt_step'))
            elif record.get('status')=='repair needed':
                record['status']='scheduled'
                record.pop('reason',None)
            elif record.get('status')=='held for review':
                record['status']='complete'
                record.pop('reason',None)
            elif record.get('status')=='waiting to join the next saved window' and not reasons:
                record['status']='scheduled'
                record.pop('reason',None)
        plans[task_id]=dict(blocked=bool(review or held_version),run=run_step,open=open_step,reasons=reasons,
                            digest=digest,version=version)
    # warnings the page shows prominently (the scheduled-results panel) until their cause clears
    in_window=(state.get('comparison_window') or {}).get('state')=='ready'
    warnings=[]
    for task in manifest.get('tasks',[]):
        record=state['tasks'].get(str(task.get('id'))) if isinstance(task,dict) else None
        if not isinstance(record,dict):
            continue
        warning=task_warning(record,in_window)
        if warning:
            record['warning']=warning
            warnings.append(warning)
        else:
            record.pop('warning',None)
    if warnings:
        state['warnings']=warnings
    else:
        state.pop('warnings',None)
    state['triggers']=dict(head=context['head'],flags=context['flags'],live_step=context['live_step'],
                           gauges=context['gauges'],read_at=time.time())
    if unnamed:
        state['manifest_errors']=[f'Ability test id {name} is not a plain artifact name; the entry never runs.' for name in unnamed]
    else:
        state.pop('manifest_errors',None)
    return plans,context


def longest_round_seconds(state, task_id):
    """The longest completed round of this task in its receipts, or of any task when it has none (0 when
    no task has one): twice it is the line past which a running test is reported as possibly stuck. The
    longest round of each task measured so far is at most 1.4 times its shortest (the design record)."""
    def longest(records):
        best=0.0
        for record in records:
            for h in (record.get('history') or []) if isinstance(record,dict) else []:
                if (isinstance(h,dict) and h.get('status')=='complete' and isinstance(h.get('started_at'),(int,float))
                        and isinstance(h.get('completed_at'),(int,float))):
                    best=max(best,float(h['completed_at'])-float(h['started_at']))
        return best
    tasks=state.get('tasks') or {}
    return longest([tasks.get(task_id)]) or longest(tasks.values())


def task_warning(record, in_window):
    """The warning a task's record carries into the status and onto the page, or None."""
    title=record.get('title') or 'A scheduled comparison'
    unmeasured=record.get('unmeasured')
    if isinstance(unmeasured,dict):
        step=unmeasured.get('step')
        return (f"{title} is unmeasured: its last two attempts" + (f" (the last at step {int(step):,})" if isinstance(step,int) else '')
                + " ended without a result, so a failed round cannot pause training, and production keeps running the "
                "mechanism it tests unchecked until a round completes. Repair the test (its script, a file it declares or "
                "its arguments) and it runs again.")
    stuck=record.get('stuck')
    if isinstance(stuck,dict):
        return (f"{title} has run {stuck.get('minutes')} minutes, past twice the longest completed round "
                f"({stuck.get('limit_minutes')} minutes); it may be stuck. Stopping it is a person's decision"
                + ("; training stays stopped while it runs inside the saved pause." if in_window else '.'))
    return None


# THE TRAINER'S SAVE PEAK BELONGS IN THE RESERVE (Oct 1 2026). With the page file at 56,000 MB a 12 GiB test
# fitted beside the running trainer under the 6 GiB reserve, but the trainer's own checkpoint save adds up to
# 12.2 GiB on the current path (docs/gain_of_function_v4/agents/save_memory_2026-10-01.md): at 10:52 AM a
# body_and_synergy test at 12 GiB plus an hourly save left 0.8 GiB of commit room (the watcher's stop), and at
# 6:13 AM a save in the same squeeze ended in a graphics-driver crash. Beside a running trainer the reserve is the
# usual 6 GiB plus that save peak; inside a saved window (trainer stopped) it stays 6 GiB. Lower SAVE_PEAK_GIB to the
# streaming save's measured peak once that save is adopted.
SAVE_PEAK_GIB = 3.2   # the streaming save (adopted Oct 1 2026, operator "keep it"): 3.112 GB of commit added at production scale; was 12.2


def beside_training_reserve(window_ready):
    return 6.0 if window_ready else 6.0 + SAVE_PEAK_GIB


def peak_private_commit_gib(children):
    """The largest peak private commit (Windows peak_pagefile; elsewhere the resident size) among an
    attempt's processes and their descendants, in GiB, or None. The venv launcher is the recorded
    child; the test runs in its child."""
    peak=None
    for child in children:
        try:
            process=psutil.Process(child['pid'])
            family=[process]+process.children(recursive=True)
        except (psutil.Error,OSError,KeyError,TypeError):
            continue
        for member in family:
            try:
                info=member.memory_info()
            except (psutil.Error,OSError):
                continue
            value=getattr(info,'peak_pagefile',None) or getattr(info,'rss',0)
            peak=max(peak or 0.0,value/2**30)
    return peak


def task_command(root, task, checkpoint, destination):
    return ([sys.executable,'-u',str(confined(root,task['script'])),
             '--checkpoint',str(checkpoint),'--out',str(destination)]
            +plain_arguments(task)+path_arguments(root,task,checkpoint))


def identity(process):
    return dict(pid=process.pid,create_time=process.create_time(),command=process.cmdline())


def command_matches(actual, expected):
    # Windows can launch the same argv through its venv redirector. The script,
    # flags, checkpoint and unique output path must still match exactly.
    return (len(actual)==len(expected) and actual[1:]==expected[1:]
            and Path(actual[0]).name.lower()==Path(expected[0]).name.lower())


def matching_children(record):
    """Recover only this attempt's exact command, including a surviving venv child.

    The unique output argument was persisted before Popen. It covers the narrow
    crash window between launching a process and writing its PID sidecar.
    """
    expected=record.get('command')
    if not expected:return [],None
    saved=record.get('child') or {}
    if saved.get('pid'):
        try:
            process=psutil.Process(saved['pid'])
            observed=identity(process)
            if (abs(observed['create_time']-float(saved['create_time']))<.001
                    and command_matches(observed['command'],expected)
                    and process.status()!=psutil.STATUS_ZOMBIE):
                return [observed],None
        except psutil.AccessDenied:
            return [],'The recorded child identity cannot currently be inspected; no replacement was launched.'
        except (psutil.NoSuchProcess,KeyError,ValueError):
            pass
    found=[]
    for process in psutil.process_iter(['pid','cmdline','create_time','status']):
        try:
            observed=process.info
            command=observed.get('cmdline') or []
            if (command_matches(command,expected)
                    and observed['create_time']>=float(record['started_at'])-2
                    and observed.get('status')!=psutil.STATUS_ZOMBIE):
                found.append(dict(pid=process.pid,create_time=observed['create_time'],command=command))
        except (psutil.NoSuchProcess,psutil.AccessDenied,KeyError):
            continue
    return sorted(found,key=lambda p:p['create_time']),None


def run_task(root, task, checkpoint, destination, command=None, run_id=None):
    """Start one child and return immediately; cycle() owns durable monitoring. With `run_id` the child's
    environment carries the row's two marks (UEMA_TEST_ROW, the task id; UEMA_TEST_RUN), by which
    test_runner.audit() matches the child and every process it starts to the task's running row."""
    command=command or task_command(root,task,checkpoint,destination)
    env=dict(os.environ,PYTHONPATH=str(root/'Source/Python'),CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2')
    if run_id:
        env.update(UEMA_TEST_ROW=str(task['id']),UEMA_TEST_RUN=str(run_id))
    log=destination.with_suffix('.log')
    with log.open('wb') as stream:
        process=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    child=identity(psutil.Process(process.pid))
    atomic_json(destination.with_suffix('.process.json'),dict(child,started_at=time.time()))
    return child


# compare_hand_body_goals_2026-09-24.py GOAL_SUITES (a test holds the two equal)
BODY_GOAL_SUITES={'hand':['curl_witness','extension_witness'],
                  'hand_and_body':['curl_witness','extension_witness','wrist_reach_witness',
                                   'supported_trunk_turn_witness','elbow_flex_witness']}
# probe_planner_candidates.py RANKING_OUTCOMES (a test holds the two equal)
PLANNER_RANKING_OUTCOMES=('goal_distance_50ms','goal_distance_end','wellness_50ms','wellness_tenure')
# fixed_probe_set_2026-10-18.py GAIN_RULE, PRIMARY_ARM and GAIN_ARMS (a test holds them equal)
FIXED_START_GAIN_RULE='movement-viewer-fixed-probe-decision-20261018'
FIXED_START_PRIMARY_ARM='direct'
FIXED_START_GAIN_ARMS=('direct','synergy')


def validate_result(payload, record):
    """A completed comparison needs its declared identity and full result shape."""
    if not isinstance(payload,dict) or not payload:
        raise ValueError('Result must be a nonempty object')
    if not isinstance(payload.get('checkpoint'),str) or not payload['checkpoint']:
        raise ValueError('Result checkpoint is missing')
    if Path(payload['checkpoint']).resolve()!=Path(record['checkpoint']).resolve():
        raise ValueError('Result checkpoint differs from the declared attempt')
    def require(condition, reason):
        if not condition:raise ValueError(reason)
    def finite(value):
        return isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value)
    require(payload.get('learning') is False,'Frozen-comparison declaration is missing')
    require(finite(payload.get('elapsed_s')) and payload['elapsed_s']>0,'Elapsed measurement time is missing')
    script=Path(record['command'][2]).name
    if script=='qualify_hand_execution_2026-09-24.py':
        require(payload.get('goals')==['curl','extension','index_selective'],'Hand goals are incomplete')
        require(payload.get('modes')==['fixed_world','moving_wrist'],'Hand comparisons are incomplete')
        require(payload.get('comparisons')==24 and payload.get('exact_replay') is True,'Hand trial count or replay verification is incomplete')
        require(payload.get('script_sha256')==record['script_sha256'],'Hand result script identity differs')
        summary=payload.get('summary',{})
        require(all(finite(summary.get(goal,{}).get(mode)) for goal in payload['goals'] for mode in payload['modes']),
                'Hand comparison summary is incomplete')
    elif script=='compare_hand_body_goals_2026-09-24.py':
        # the goal suite the scheduler asked for (manifest 'arguments': --goal-suite), 'hand' without it
        command=record.get('command') or []
        suite=command[command.index('--goal-suite')+1] if '--goal-suite' in command[:-1] else 'hand'
        goals=BODY_GOAL_SUITES.get(suite)
        require(payload.get('bodies')==8 and goals is not None and payload.get('goals')==goals
                and payload.get('goal_suite')==suite,'Body comparison scope is incomplete')
        require(payload.get('weights_unchanged') is True and payload.get('exact_replay') is True,'Body state verification is incomplete')
        require(payload.get('script_sha256')==record['script_sha256'],'Body result script identity differs')
        summary=payload.get('summary',{})
        for metric in ('physical_goal_error','hand_joint_error_deg','wrist_goal_error_cm'):
            row=summary.get(metric,{})
            require(all(finite(row.get('means',{}).get(mode)) for mode in ('controller','direct','synergy')),
                    'Body comparison means are incomplete')
            interval=row.get('paired_bootstrap95',[])
            require(len(interval)==2 and all(finite(v) for v in interval),'Body comparison uncertainty is incomplete')
        # the outside review's statistic (2026-09-25): pooled, and per goal (a goal whose arms ended
        # exactly alike, or differed in only one body, has no interval and says 'undetermined')
        from .command_ranking import valid_block
        ranking=payload.get('ranking') if isinstance(payload.get('ranking'),dict) else {}
        by_goal=ranking.get('by_goal') if isinstance(ranking.get('by_goal'),dict) else {}
        require(valid_block(ranking.get('pooled'),allow_undetermined=True)
                and all(valid_block(by_goal.get(goal),allow_undetermined=True) for goal in goals),
                'Body comparison ranking is incomplete')
        goal_rows=payload.get('by_goal') if isinstance(payload.get('by_goal'),dict) else {}
        for goal in goals:
            row=goal_rows.get(goal) if isinstance(goal_rows.get(goal),dict) else {}
            check=row.get('witness_check') if isinstance(row.get('witness_check'),dict) else {}
            count=check.get('informative_bodies')
            require(check.get('bodies')==8 and isinstance(count,int) and not isinstance(count,bool) and 0<=count<=8
                    and all(finite(check.get(key)) and check[key]>=0
                            for key in ('requested_raw','requested_executed','separation_from_controller','moved_from_start'))
                    and (check.get('achieved_share') is None or finite(check['achieved_share'])),
                    'Body comparison witness check is incomplete')
    elif script=='probe_saved_gaze.py':
        require(payload.get('script_sha256')==record['script_sha256'],'Eye result script identity differs')
        require(payload.get('bodies')==8 and payload.get('evaluated_bodies')==8
                and payload.get('commands')==['hold','left','right','up','down']
                and payload.get('starts_per_body')==2,'Eye comparison scope is incomplete')
        require(payload.get('command_branches')==80 and payload.get('completed_command_branches')==80
                and payload.get('replays_per_branch')==2,'Eye trial count is incomplete')
        require(payload.get('exact_replay') is True and payload.get('all_candidate_traces_replayed_exactly') is True
                and payload.get('weights_unchanged') is True,'Eye state verification is incomplete')
        require(payload.get('gaze_reflex') is False and payload.get('spontaneous_saccades') is False,
                'Eye commands may have been overridden')
        require(all(isinstance(payload.get(key),dict) and payload[key]
                    for key in ('helper_sha256','source_sha256','episode'))
                and bool(payload.get('runtime_sha256')),'Eye source identity is incomplete')
        for start in ('saved_starts','after_head_command'):
            for horizon in ('50ms','100ms','150ms'):
                for metric in ('eye_prediction_error_degrees','visual_prediction_mse'):
                    row=payload.get('summary',{}).get(start,{}).get(horizon,{}).get(metric,{})
                    require(row.get('bodies')==8 and all(finite(row.get(key)) and row[key]>=0
                            for key in ('candidate_mean','comparison_mean')),'Eye comparison means are incomplete')
                    interval=row.get('paired95_difference',[])
                    require(len(interval)==2 and all(finite(v) for v in interval) and interval[0]<=interval[1],
                            'Eye comparison uncertainty is incomplete')
    elif script=='probe_saved_self_touch_choices.py':
        require(isinstance(payload.get('script_sha256'),str) and payload['script_sha256']==record['script_sha256'],
                'Self-touch result script identity is missing or differs')
        if payload.get('probe')=='self_touch_layers_v4':
            # the three-layer form (2026-09-25, docs/gain_of_function_v4/agents/self_touch_layers.md):
            # reachability, prediction (whether and where) and selection reported separately. Every judged
            # score's sign rule is recomputed here: a body votes with its share minus 50 once it has at least 5
            # judged pairs from at least 2 touch events; at least 5 voting bodies; one-sided p at most 0.05. The
            # pass uses the touches of a hand on another body area.
            require(payload.get('bodies')==8 and payload.get('starts_per_body')==3 and payload.get('evaluated_starts')==24,
                    'Self-touch start count is incomplete')
            require(payload.get('weights_unchanged') is True and payload.get('all_candidate_traces_replayed_exactly') is True,
                    'Self-touch state verification is incomplete')
            require(finite(payload.get('tie_band')) and payload['tie_band']>0,'Self-touch tie band is missing')
            layers=payload.get('layers') if isinstance(payload.get('layers'),dict) else {}
            reach,prediction,selection=(layers.get(k) if isinstance(layers.get(k),dict) else {}
                                        for k in ('reachability','prediction','selection'))
            def count(value,top):
                return isinstance(value,int) and not isinstance(value,bool) and 0<=value<=top
            kinds={'hand_to_body':'hand_to_body_','any_touch':''}
            offered,consistent={},True
            for h in ('50ms','tenure'):
                for kind,label in kinds.items():
                    rs,rb=reach.get('reachable_starts_'+label+h),reach.get('reachable_bodies_'+label+h)
                    ps,pb=reach.get('starts_offering_a_'+label+'pair_'+h),reach.get('bodies_offering_a_'+label+'pair_'+h)
                    ok=bool(count(rs,24) and count(rb,8) and count(ps,24) and count(pb,8)
                            and ps<=rs and pb<=rb and rb<=rs and pb<=ps)
                    consistent=consistent and ok
                    if ok and h=='50ms':
                        offered[kind]=(ps,pb)
                # a hand touching another body area is a touch: never more of the first than of every touch
                consistent=consistent and bool(
                    reach['reachable_starts_hand_to_body_'+h]<=reach['reachable_starts_'+h]
                    and reach['starts_offering_a_hand_to_body_pair_'+h]<=reach['starts_offering_a_pair_'+h])
            require(consistent and len(offered)==2,'Self-touch reachability counts are missing or inconsistent')
            bodies={str(b) for b in range(8)}
            def tail(n,k):
                return sum(math.comb(n,i) for i in range(k,n+1))/2.**n
            def measured(stat,low,high):
                # a pooled value: none without a body, no interval with one body, else an interval holding it
                if not isinstance(stat,dict) or not count(stat.get('bodies'),8):
                    return False
                value,interval=stat.get('value'),stat.get('interval')
                if stat['bodies']==0:
                    return value is None and interval is None
                if not (finite(value) and low<=value<=high):
                    return False
                if stat['bodies']==1:
                    return interval is None
                return (isinstance(interval,list) and len(interval)==2 and all(finite(v) for v in interval)
                        and low-1e-9<=interval[0]<=value+1e-9 and value-1e-9<=interval[1]<=high+1e-9)
            def pairs_of(stat,top):
                pairs=stat.get('pairs')
                return count(pairs,top) and (pairs==0)==(stat['bodies']==0)
            def regions_ok(table,total=None):
                return (isinstance(table,dict) and all(isinstance(k,str) and count(v,10**9) for k,v in table.items())
                        and (total is None or sum(table.values())==total))
            def judged_ok(stat,top_pairs,top_bodies):
                # a judged share: its pooled value over its pairs equal to the pair-weighted mean of its per-body
                # shares, inside its body-resampled interval; the goal line and the distance to it; the voters
                # (exactly the bodies with at least 5 judged pairs from at least 2 touch events), each voting with
                # its share minus 50; the count above, the exact tail probability and the verdict they give
                if not (measured(stat,0.,1.) and pairs_of(stat,top_pairs) and stat['bodies']<=top_bodies):
                    return False
                distance=stat.get('distance_to_goal_line')
                if not (stat.get('chance')==.5 and stat.get('goal_line')==.8
                        and (stat.get('value') is None)==(distance is None)
                        and (distance is None or (finite(distance) and abs(distance-(.8-stat['value']))<=1e-9))):
                    return False
                pairs,events,shares=stat.get('per_body_pairs'),stat.get('per_body_events'),stat.get('per_body')
                if not (isinstance(pairs,dict) and isinstance(events,dict) and isinstance(shares,dict)
                        and set(pairs)==set(events)==set(shares) and set(pairs)<=bodies and len(pairs)==stat['bodies']):
                    return False
                if not all(count(pairs[b],10**9) and pairs[b]>=1 and count(events[b],pairs[b]) and events[b]>=1
                           and finite(shares[b]) and 0<=shares[b]<=1 for b in pairs):
                    return False
                if sum(pairs.values())!=stat['pairs']:
                    return False
                if pairs and abs(stat['value']-sum(shares[b]*pairs[b] for b in pairs)/stat['pairs'])>1e-9:
                    return False
                votes=stat.get('votes') if isinstance(stat.get('votes'),dict) else {}
                voters=votes.get('voters')
                if not isinstance(voters,dict) or (votes.get('alpha'),votes.get('min_pairs_per_body'),
                        votes.get('min_events_per_body'),votes.get('min_voting_bodies'))!=(.05,5,2,5):
                    return False
                if set(voters)!={b for b in pairs if pairs[b]>=5 and events[b]>=2}:
                    return False
                if not all(finite(voters[b]) and abs(voters[b]-(shares[b]-.5))<=1e-9 for b in voters):
                    return False
                n,k=len(voters),sum(1 for v in voters.values() if v>0)
                p=votes.get('p_value')
                if (votes.get('bodies_voting'),votes.get('bodies_above'))!=(n,k):
                    return False
                if (n==0)!=(p is None) or (n and not (finite(p) and abs(p-tail(n,k))<=1e-12)):
                    return False
                verdict=stat.get('verdict')
                if not isinstance(verdict,str):
                    return False
                if stat['bodies']==0:
                    return verdict.startswith('not measured: ')
                if n<5:
                    return verdict.startswith('not decided: ')
                return verdict==('above its control' if p<=.05 else 'not above its control')
            def prediction_ok(block,starts,bodies_offered):
                # every pair (reported, not judged); 'whether' on the pairs in which the toucher asked no handle
                # for a bigger move and 'where' on crossed pairs, each judged; the block passes only when both do
                if not isinstance(block,dict):
                    return False
                every,whether,where=(block.get(k) for k in ('all_pairs','whether','where'))
                if not (measured(every,0.,1.) and every.get('judged') is False and pairs_of(every,10**9)
                        and (every.get('starts'),every['bodies'])==(starts,bodies_offered)
                        and regions_ok(every.get('touch_region_pairs'))):
                    return False
                if not (judged_ok(whether,every['pairs'],every['bodies']) and regions_ok(whether.get('touch_region_pairs'))
                        and judged_ok(where,10**9,every['bodies']) and regions_ok(where.get('touch_region_pairs'))
                        and count(where.get('unresolved_crossed_pairs'),10**9)):
                    return False
                verdicts=[whether['verdict'],where['verdict']]
                verdict=block.get('verdict')
                if not isinstance(verdict,str):
                    return False
                if every['bodies']==0:
                    return verdict.startswith('not measured: ')
                if verdicts==['above its control']*2:
                    return verdict=='above its control'
                if 'not above its control' in verdicts:
                    return verdict=='not above its control'
                return verdict.startswith('not decided: ')
            def selection_ok(stat,starts,bodies_offered):
                # the planner's objective against each touching command's matched command, judged against 50
                if not (judged_ok(stat,10**9,bodies_offered) and count(stat.get('starts'),starts)
                        and stat.get('tie_band')==payload['tie_band'] and regions_ok(stat.get('touch_region_pairs'))):
                    return False
                loose=stat.get('unfiltered')
                if not (measured(loose,0.,1.) and loose.get('judged') is False and pairs_of(loose,10**9)):
                    return False
                moves=stat.get('move_relation')
                if not (isinstance(moves,dict) and set(moves)=={'smaller','bigger','equal','mixed'}
                        and all(count(v,10**9) for v in moves.values()) and sum(moves.values())==stat['pairs']):
                    return False
                gaps=stat.get('wellness_gap_abs')
                return gaps is None or (isinstance(gaps,dict) and all(finite(gaps.get(k)) and gaps[k]>=0
                                                                      for k in ('median','p90','max'))
                                        and gaps['median']<=gaps['p90']+1e-12 and gaps['p90']<=gaps['max']+1e-12)
            names=('hand_to_body','any_touch')
            routes=prediction.get('routes') if isinstance(prediction.get('routes'),dict) else {}
            require(all(prediction_ok(prediction.get(kind),*offered[kind]) for kind in names)
                    and all(prediction_ok(routes.get(route),*offered['any_touch'])
                            for route in ('forward_model_only','order_input_only'))
                    and prediction.get('verdict')==prediction['hand_to_body']['verdict'],
                    'Self-touch prediction statistics are incomplete or inconsistent')
            require(all(selection_ok(selection.get(kind),*offered[kind])
                        and selection[kind]['bodies']<=prediction[kind]['all_pairs']['bodies'] for kind in names)
                    and selection.get('verdict')==selection['hand_to_body']['verdict'],
                    'Self-touch selection statistics are incomplete or inconsistent')
            above=lambda block:block['verdict']=='above its control'
            require(isinstance(payload.get('passed'),bool) and payload['passed']==(
                    above(prediction['hand_to_body']) and above(selection['hand_to_body'])),
                    'Self-touch outcome disagrees with its declared rule')
        elif payload.get('probe') is not None:
            raise ValueError('Self-touch result version %r is not one this validator knows' % (payload.get('probe'),))
        else:
            require(payload.get('bodies')==8 and payload.get('evaluated_held_out_trials')==16,'Self-touch trial count is incomplete')
            require(payload.get('weights_unchanged') is True and payload.get('all_candidate_traces_replayed_exactly') is True,
                    'Self-touch state verification is incomplete')
            require(isinstance(payload.get('passed_private_comparison'),bool),'Self-touch decision is missing')
            for name in ('controller','hold','mean_exploration'):
                row=payload.get('comparisons',{}).get(name,{})
                require(isinstance(row.get('bodies'),int) and 0<=row['bodies']<=8
                        and isinstance(row.get('trials'),int) and 0<=row['trials']<=16,
                        'Self-touch paired comparison scope is missing')
                if row['trials']==0:
                    require(row.get('improvement_above_uncertainty') is False,'A no-choice comparison cannot claim improvement')
                    continue
                for field in ('selected_success_fraction','comparison_success_fraction'):
                    require(finite(row.get(field)) and 0<=row[field]<=1,'Self-touch outcome fraction is invalid')
                interval=row.get('difference_percentage_points_ci95',[])
                require(len(interval)==2 and all(finite(v) for v in interval),'Self-touch comparison uncertainty is incomplete')
    elif script=='probe_planner_candidates.py':
        # docs/gain_of_function_v4/probe_planner_candidates.py: why the planner's choices tie
        require(isinstance(payload.get('script_sha256'),str) and payload['script_sha256']==record['script_sha256'],
                'Planner-candidate result script identity is missing or differs')
        # v3 (2026-10-02): depths 2 and 3 read each candidate's executed pose, and the second pass must reproduce
        # the first pass's depth-1 prediction (probe_planner_candidates DEPTH_POSE, EXECUTED_POSE_DEPTH1_TOLERANCE)
        require(payload.get('probe')=='planner_candidates_v3','Planner-candidate probe version differs')
        require(payload.get('depth_pose')=='executed','The planner-candidate depths do not read the executed pose')
        require(finite((payload.get('route_checks_max') or {}).get('executed_pose_depth1_max_abs'))
                and payload['route_checks_max']['executed_pose_depth1_max_abs']<=1e-6,
                'The executed-pose pass did not reproduce the depth-1 prediction')
        require(payload.get('bodies')==8 and payload.get('starts_per_body')==2 and payload.get('evaluated_starts')==16,
                'Planner-candidate start count is incomplete')
        require(payload.get('candidates')==['chosen','hold','reflex','perturb_1','perturb_2']
                and payload.get('depths')==[1,2,3],'Planner-candidate scope is incomplete')
        require(isinstance(payload.get('tenure_steps'),int) and payload['tenure_steps']>=3,
                'The order tenure is shorter than the depth comparison')
        require(payload.get('weights_unchanged') is True and payload.get('exact_replay') is True
                and payload.get('all_candidate_traces_replayed_exactly') is True
                and isinstance(payload.get('observation_replay_checks'),int) and payload['observation_replay_checks']>=8,
                'Planner-candidate state verification is incomplete')
        verdict=payload.get('verdict',{})
        require(all(isinstance(verdict.get(k),bool) for k in ('forward_model_flat','evaluator_flat',
                'planner_ignores_predicted_wellness','depth3_better_physical_decision')),
                'Planner-candidate verdict is incomplete')
        require(payload.get('passed') is verdict['depth3_better_physical_decision'],
                'Planner-candidate outcome disagrees with its verdict')
        routes=payload.get('route_checks_max',{})
        require(all(finite(routes.get(k)) and 0<=routes[k]<=1e-3 for k in
                    ('record_pred_wellness_max_abs','record_goal_distance_max_abs','alternatives_execution_max_abs_cm')),
                'The probe did not reproduce the production planning record')
        summary=payload.get('summary',{})
        def interval_ok(stat):
            interval=(stat or {}).get('interval')
            return (isinstance(interval,list) and len(interval)==2 and all(finite(v) for v in interval)
                    and interval[0]<=interval[1])
        for stat in (summary.get('evaluator',{}).get('transfer'),summary.get('forward_model',{}).get('transfer_judge_units'),
                     summary.get('chain',{}).get('transfer'),summary.get('planner',{}).get('chosen_rank_by_predicted_wellness')):
            require(finite((stat or {}).get('value')) and interval_ok(stat),
                    'A planner-candidate transfer or its uncertainty is incomplete')
        band=summary.get('tie_band',{})
        require(finite(band.get('remeasured_planner_route')) and finite(band.get('evaluator_on_actual_outcomes')),
                'The re-measured tie band is incomplete')
        for selector in ('wellness','objective'):
            row=summary.get('depth',{}).get(selector,{}).get('wellness_tenure_depth3_minus_depth1',{})
            require(row.get('bodies')==8 and finite(row.get('mean')) and interval_ok(row),
                    'The depth comparison is incomplete')
        # the outside review's statistic (planner_candidates_v2): every outcome's pairwise ranking
        from .command_ranking import valid_block
        ranking=summary.get('ranking') if isinstance(summary.get('ranking'),dict) else {}
        require(all(valid_block(ranking.get(name)) for name in PLANNER_RANKING_OUTCOMES),
                'The planner-candidate ranking is incomplete')
    elif script=='compare_hand_memory_route_2026-09-24.py':
        # docs/gain_of_function_v3/compare_hand_memory_route_2026-09-24.py: the learner's remembered
        # hand-goal stage on against off, at two motor-latch decisions of each held-out body 4-7
        require(payload.get('script_sha256')==record['script_sha256'],'Hand-memory result script identity differs')
        collected=payload.get('collected')
        require(payload.get('held_out_bodies')==[4,5,6,7] and payload.get('starts_per_body')==2
                and payload.get('completed_trials')==8 and isinstance(collected,dict) and len(collected)==4
                and all(isinstance(v,dict) and v.get('starts')==2 for v in collected.values()),
                'Hand-memory trial count is incomplete')
        require(payload.get('weights_unchanged') is True and payload.get('exact_replay') is True
                and payload.get('nonfinger_commands_unchanged') is True and payload.get('sensory_goals_identical') is True
                and payload.get('production_modified') is False,'Hand-memory state verification is incomplete')
        overrides=payload.get('overrides') or {}
        require(overrides.get('hand_goals')=='memory' and overrides.get('hand_target_mode')=='wrist_relative_v1',
                'Hand-memory route settings differ')
        command=record.get('command') or []
        calibration=Path(command[command.index('--calibration')+1]) if '--calibration' in command[:-1] else None
        require(calibration is not None and calibration.is_file()
                and payload.get('calibration_file_sha256')==hashlib.sha256(calibration.read_bytes()).hexdigest(),
                'Hand-memory calibration identity differs from the declared calibration file')
        summary=payload.get('summary',{})
        for metric in ('hand_shape_error','finger_position_error_cm','finger_orientation_error_deg'):
            row=summary.get(metric,{})
            require(all(finite(row.get('means',{}).get(mode)) for mode in ('baseline','hand_memory')),
                    'Hand-memory comparison means are incomplete')
            interval=row.get('paired95_difference',[])
            require(len(interval)==2 and all(finite(v) for v in interval) and interval[0]<=interval[1],
                    'Hand-memory comparison uncertainty is incomplete')
            require(isinstance(row.get('bodies_improved'),int) and not isinstance(row['bodies_improved'],bool)
                    and 0<=row['bodies_improved']<=4,'Hand-memory improved-body count is invalid')
        main=summary['hand_shape_error']
        require(isinstance(payload.get('passed'),bool)
                and payload['passed']==(main['bodies_improved']==4 and main['paired95_difference'][1]<0),
                'Hand-memory outcome disagrees with its declared admission rule')
    elif script=='fixed_probe_set_2026-10-18.py':
        # docs/gain_of_function_v3/fixed_probe_set_2026-10-18.py (Oct 18, 2026): the body comparison from the frozen
        # set's starts and goals, task fixed_start_body_and_synergy. The frozen set and the anchor the scheduler passed
        # (path_arguments --frozen-set-receipt and --anchor) must be the ones the result names, the result's mode the one
        # its arguments ask for, and every restored start must equal the set's start qpos; then the body comparison's own
        # checks; then the gain object against the bound reference (--reference-receipt), below.
        command=record.get('command') or []
        def argument(flag):
            return command[command.index(flag)+1] if flag in command[:-1] else None
        receipt_path,anchor_path=argument('--frozen-set-receipt'),argument('--anchor')
        receipt=read(receipt_path) if receipt_path else {}
        goals=BODY_GOAL_SUITES['hand_and_body']
        require(payload.get('mode')==('frozen_start_anchored' if anchor_path else 'frozen_start')
                and payload.get('probe')=='fixed_start_body_goals_v2',
                'Frozen-start declaration is missing or names another mode than its arguments')
        require(payload.get('bodies')==8 and payload.get('goals')==goals and payload.get('goal_suite')=='hand_and_body',
                'Frozen-start comparison scope is incomplete')
        frozen=payload.get('frozen_set') if isinstance(payload.get('frozen_set'),dict) else {}
        require(isinstance(receipt.get('set_sha256'),str) and frozen.get('sha256')==receipt['set_sha256'],
                'Frozen set identity differs from the declared receipt')
        if anchor_path:
            anchor=payload.get('anchor') if isinstance(payload.get('anchor'),dict) else {}
            require(Path(anchor_path).is_file()
                    and anchor.get('sha256')==hashlib.sha256(Path(anchor_path).read_bytes()).hexdigest(),
                    'Anchor identity differs from the declared anchor file')
        require(payload.get('starts_equal_to_set') is True,'A restored start differs from the frozen set')
        require(payload.get('weights_unchanged') is True and payload.get('exact_replay') is True,
                'Frozen-start state verification is incomplete')
        require(payload.get('script_sha256')==record['script_sha256'],'Frozen-start result script identity differs')
        summary=payload.get('summary',{})
        for metric in ('physical_goal_error','hand_joint_error_deg','wrist_goal_error_cm'):
            row=summary.get(metric,{})
            require(all(finite(row.get('means',{}).get(mode)) for mode in ('controller','direct','synergy')),
                    'Frozen-start comparison means are incomplete')
            interval=row.get('paired_bootstrap95',[])
            require(len(interval)==2 and all(finite(v) for v in interval),'Frozen-start comparison uncertainty is incomplete')
        against=summary.get('physical_goal_error',{}).get('against_controller',{})
        require(all(len((against.get(k) or {}).get('paired_bootstrap95') or [])==2
                    and all(finite(v) for v in against[k]['paired_bootstrap95']) for k in ('direct','synergy')),
                'Frozen-start planner-against-controller intervals are incomplete')
        from .command_ranking import valid_block
        ranking=payload.get('ranking') if isinstance(payload.get('ranking'),dict) else {}
        by_goal=ranking.get('by_goal') if isinstance(ranking.get('by_goal'),dict) else {}
        require(valid_block(ranking.get('pooled'),allow_undetermined=True)
                and all(valid_block(by_goal.get(goal),allow_undetermined=True) for goal in goals),
                'Frozen-start comparison ranking is incomplete')
        goal_rows=payload.get('by_goal') if isinstance(payload.get('by_goal'),dict) else {}
        for goal in goals:
            row=goal_rows.get(goal) if isinstance(goal_rows.get(goal),dict) else {}
            check=row.get('witness_check') if isinstance(row.get('witness_check'),dict) else {}
            count=check.get('informative_bodies')
            require(check.get('bodies')==8 and isinstance(count,int) and not isinstance(count,bool) and 0<=count<=8
                    and all(finite(check.get(key)) and check[key]>=0
                            for key in ('requested_raw','requested_executed','separation_from_controller','moved_from_start'))
                    and (check.get('achieved_share') is None or finite(check['achieved_share'])),
                    'Frozen-start comparison witness check is incomplete')
        # THE GAIN RULE (declared_line). The reference the result was compared with must be the one the manifest binds: the
        # receipt the command names (--reference-receipt), whose bytes hash to the SHA-256 the command declares
        # (--reference-receipt-sha256, a plain argument of the manifest), never a folder chosen at run time. A comparison
        # that is not valid is refused here, so the attempt fails (INVALID in its registry row) and follows the retry rules;
        # a valid one must carry both arms' arithmetic consistent with their lines, the changed controls and the label.
        reference_path,reference_sha=argument('--reference-receipt'),argument('--reference-receipt-sha256')
        require(bool(reference_path) and bool(reference_sha) and Path(reference_path).is_file()
                and hashlib.sha256(Path(reference_path).read_bytes()).hexdigest()==reference_sha,
                'The reference receipt the command names is missing or differs from its declared SHA-256')
        bound=read(reference_path)
        files=bound.get('files') if isinstance(bound.get('files'),dict) else {}
        bound_anchor=bound.get('anchor') if isinstance(bound.get('anchor'),dict) else {}
        own_anchor=payload.get('anchor') if isinstance(payload.get('anchor'),dict) else {}
        require((bound.get('frozen_set') or {}).get('sha256')==frozen.get('sha256') and bound.get('mode')==payload.get('mode')
                and bound_anchor.get('sha256')==own_anchor.get('sha256'),
                'The bound reference ran under other search conditions (frozen set, mode or anchor)')
        gain=payload.get('gain') if isinstance(payload.get('gain'),dict) else {}
        named=gain.get('reference') if isinstance(gain.get('reference'),dict) else {}
        receipt_named=named.get('receipt') if isinstance(named.get('receipt'),dict) else {}
        compared=(gain.get('this') or {}).get('checkpoint') if isinstance(gain.get('this'),dict) else None
        require(gain.get('rule_version')==FIXED_START_GAIN_RULE and gain.get('primary_arm')==FIXED_START_PRIMARY_ARM,
                'Fixed-start gain object is missing or of another rule')
        require(receipt_named.get('sha256')==reference_sha
                and named.get('sha256')==(files.get('result.json') or {}).get('sha256')
                and named.get('trials_sha256')==(files.get('trials.jsonl') or {}).get('sha256')
                and named.get('checkpoint_step')==bound.get('checkpoint_step')
                and isinstance(compared,str) and Path(compared).resolve()==Path(payload['checkpoint']).resolve(),
                'The gain names another reference or checkpoint than the bound ones')
        require(gain.get('status') in ('valid','invalid'),'Fixed-start gain status is missing')
        if gain['status']=='invalid':
            reasons=gain.get('invalid_reasons')
            raise ValueError('The comparison with the bound reference is invalid: '
                             +('; '.join(str(r) for r in reasons) if isinstance(reasons,list) and reasons
                               else 'no reason recorded'))
        require(not gain.get('invalid_reasons'),'A valid fixed-start gain carries invalid reasons')
        arms=gain.get('arms') if isinstance(gain.get('arms'),dict) else {}
        def interval_ok(block):
            interval=block.get('paired_bootstrap95') if isinstance(block,dict) else None
            return (isinstance(interval,list) and len(interval)==2 and all(finite(v) for v in interval)
                    and interval[0]<=interval[1] and finite(block.get('difference_mean')))
        for arm in FIXED_START_GAIN_ARMS:
            block=arms.get(arm) if isinstance(arms.get(arm),dict) else {}
            per_goal=block.get('per_goal') if isinstance(block.get('per_goal'),dict) else {}
            per_body=block.get('per_body') if isinstance(block.get('per_body'),list) else []
            require(interval_ok(block) and block.get('bodies')==8 and len(per_body)==8 and set(per_goal)==set(goals)
                    and all(interval_ok(per_goal[g]) for g in goals)
                    and block.get('line_met') is (block['paired_bootstrap95'][1]<0)
                    and all(interval_ok(block.get(metric)) for metric in ('hand_joint_error_deg','wrist_goal_error_cm')),
                    'Fixed-start gain arithmetic of the %s arm is incomplete or disagrees with its line'%arm)
        require(gain.get('line_met') is arms[FIXED_START_PRIMARY_ARM]['line_met'],
                "The fixed-start gain line is not the direct arm's")
        require(isinstance(gain.get('changed_controls'),dict) and isinstance(gain.get('label'),str) and gain['label'].strip(),
                'Fixed-start gain comparison lacks its changed controls or label')
    else:
        raise ValueError('No result validator is declared for this comparison script')


def finished_result(record):
    destination=Path(record['output'])
    result=destination/'result.json'
    if not result.is_file():
        return dict(status='failed',log=str(destination.with_suffix('.log')),
                    reason='The child exited or was interrupted without a complete result.')
    try:
        payload=read(result)
        validate_result(payload,record)
    # AttributeError and IndexError too: a malformed field (a list where an object belongs) must fail
    # the attempt, never escape cycle() and leave the task 'running' on every later cycle
    except (OSError,ValueError,TypeError,KeyError,AttributeError,IndexError) as error:
        return dict(status='failed',reason=f'Unreadable or mismatched result: {error}',result=str(result))
    try:
        line=declared_line(Path(record['command'][2]).name,payload)
    except (TypeError,KeyError,AttributeError,IndexError,ValueError):
        line=None
    return dict(status='complete',result=str(result),
                result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),
                outcome=payload.get('passed_private_comparison',payload.get('passed')),line_met=line)


def reconcile_task(state, task_id, root=None):
    """Refresh one active attempt, without requiring the original parent process. With `root` a finished attempt
    is written into its test registry row (registry_finished)."""
    record=state['tasks'][task_id]
    children,error=matching_children(record)
    if error:
        record.update(status='process identity unavailable',reason=error)
        state.update(state='waiting for process identity',active_task=task_id)
        return
    if children:
        record.update(status='running',child=children[0],observed_children=children,last_seen_at=time.time())
        if root is not None and record.get('registry_run_id') and 'registry_start' not in record:
            # a launch whose start was not written (the scheduler ended between starting the child and the write)
            why=registry_started(root,task_id,record,children[0])
            record['registry_start']='running' if why is None else f'not written: {why}'
        # the round's memory, measured (only the planner probe recorded its own before 2026-09-25)
        peak=peak_private_commit_gib(children)
        if peak is not None:
            record['peak_private_commit_gib']=max(float(record.get('peak_private_commit_gib') or 0.0),peak)
        # A PAUSE ALWAYS ENDS OR IS REPORTED: a test running past twice its longest completed round (of
        # any task, when it has none) is reported; it is never killed, that stays a person's call
        limit=2*longest_round_seconds(state,task_id)
        elapsed=time.time()-float(record.get('started_at') or time.time())
        if limit and elapsed>limit:
            record['stuck']=dict(minutes=round(elapsed/60,1),limit_minutes=round(limit/60,1))
        else:
            record.pop('stuck',None)
        state.update(state='running',active_task=task_id)
        return
    previous_line=record.get('line_met')
    outcome=finished_result(record)
    if outcome['status']=='failed' and refused_as_stale(record):
        # counted as a failed attempt all the same (two hold the test and release the window); a
        # restarted scheduler takes it back (forgive_stale_refusals)
        outcome['reason']=('The probe refused to run under a scheduler that started before the last change to its '
                           'validator; restart the ability scheduler and it runs.')
    record.update(outcome,completed_at=time.time())
    if root is not None and record.get('registry_run_id'):
        try:
            why,written=registry_finished(root,task_id,record,outcome)
        except Exception as error:       # the row's write never costs the scheduler its cycle
            why,written=f'{type(error).__name__}: {error}',None
        record['registry_end']=written if why is None else f'not written: {why}'
    if outcome['status']=='complete':
        record['completed_step']=record['attempt_step']
        record.update(failed_version=None,failures_in_a_row=0)
        # what the round measured becomes the basis its triggers compare against (plan_tasks)
        if isinstance(record.get('launch_basis'),dict):
            record['basis']=record.pop('launch_basis')
        record.pop('triggered',None)
        line=outcome.get('line_met')
        if isinstance(line,bool) and isinstance(previous_line,bool) and line!=previous_line:
            record['line_changed']=dict(step=record['attempt_step'],at=record['completed_at'],before=previous_line,after=line)
        if line is None:
            record['line_met']=previous_line
    elif record.get('test_version'):
        # consecutive failures of one test version, across checkpoints (test_version)
        if record.get('failed_version')==record['test_version']:
            record['failures_in_a_row']=int(record.get('failures_in_a_row',0))+1
        else:
            record.update(failed_version=record['test_version'],failures_in_a_row=1)
    record.pop('stuck',None)
    if outcome['status']=='complete':
        record.pop('unmeasured',None)
    elif record.get('hold_on_failed_comparison') and int(record.get('failures_in_a_row',0))>=2:
        # THE COORDINATOR'S DECISION OF 2026-09-25 (review gap a): two attempts of a mechanism that steers
        # production ending without a result are no evidence that the mechanism fails, so training is
        # not held; the mechanism is unmeasured, and a warning stays in the status and on the page until
        # a round completes
        record.setdefault('unmeasured',dict(since=record['completed_at'],step=record.get('attempt_step')))
    receipt={k:record.get(k) for k in ('status','attempt_step','attempts','script_sha256','test_version',
            'failures_in_a_row','checkpoint','started_at','completed_at','output','command','child','result',
            'result_sha256','outcome','reason','peak_private_commit_gib')}
    # this attempt's own declared line (the record keeps the last completed one between rounds)
    receipt['line_met']=outcome.get('line_met') if outcome['status']=='complete' else None
    atomic_json(Path(record['output']).with_suffix('.receipt.json'),receipt)
    record.setdefault('history',[]).append(receipt)
    # A MECHANISM THAT STEERS PRODUCTION HOLDS TRAINING WHEN ITS ROUND FAILS (the operator, 2026-09-25
    # 9:09 AM PDT): a completed round whose declared comparison did not pass keeps the saved window's
    # pause for review (ability_comparison_window.advance). A round that ended without a valid result is
    # a failed attempt and follows the retry rules instead: it did not measure the mechanism.
    if outcome['status']=='complete' and record.get('hold_on_failed_comparison') and outcome.get('outcome') is False:
        record['review_hold']=dict(
            step=record['attempt_step'],result=record.get('result'),at=record['completed_at'],title=record.get('title'),
            reason=(f"{record.get('title') or task_id}: the round at step {int(record['attempt_step']):,} did not pass its "
                    f"declared comparison. {record['hold_on_failed_comparison']} Training stays paused for review."))
        record.update(status='held for review',reason=record['review_hold']['reason'])
    state.update(state='waiting',active_task=None)


def cycle(root, manifest_path, folder):
    from .commit_guard import commit_reading, verdict
    from . import ability_comparison_window as comparison_window
    manifest=read(manifest_path)
    status_path=folder/'status.json'
    state=read(status_path)
    state.setdefault('tasks',{})
    state.update(pid=os.getpid(),timestamp=time.time(),manifest=str(manifest_path),state='waiting')
    supervisor=read(root/'Saved/mimo_bridge/training_continuation.json')
    run=Path(supervisor.get('active_run',''))
    # every task's schedule, triggers and hold, written before attempts are recovered or a window
    # advances: the window's pending list and the dashboard read the statuses this writes
    plans,context=plan_tasks(root,manifest,state,run,scheduler_is_current())
    # Recover prior attempts before applying launch gates, even during a repair
    # pause or lineage change. A surviving private child must not be duplicated.
    for task_id,record in state['tasks'].items():
        if record.get('status') in ACTIVE_STATES:
            reconcile_task(state,task_id,root)
            atomic_json(status_path,state)
            return state

    def runnable(step):
        """The tasks that can still run on a window's checkpoint at `step` this cycle: due there, not
        held, not already done there, attempts left there, memory not already failed there. The window
        waits for exactly these (ability_comparison_window.advance)."""
        out=set()
        for task in manifest.get('tasks',[]):
            plan=plans.get(str(task.get('id'))) if isinstance(task,dict) else None
            if not plan or plan['blocked'] or plan['run']>step:
                continue
            record=state['tasks'].get(task['id']) or {}
            if int(record.get('completed_step',-1))>=step:
                continue
            if (record.get('attempt_step')==step and record.get('test_version')==plan['version']
                    and int(record.get('attempts',0))>=2):
                continue
            if record.get('status')=='memory capacity needs repair' and record.get('memory_window_step')==step:
                continue
            out.add(task['id'])
        return out
    window_ready=comparison_window.advance(root,folder,state,run,runnable)
    reason=pause_reason(root/'Saved/mimo_bridge',run)
    if reason and not window_ready:
        window=state.get('comparison_window',{})
        if window.get('state')=='held for review':
            state['state']='training held for review'
            reason=str(window.get('reason') or reason)
        else:
            state['state']=('saving training before scheduled comparisons' if window.get('state')=='saving'
                            else 'production paused for repair')
        state['pause_reason']=reason
        atomic_json(status_path,state)
        return state
    state.pop('pause_reason',None)
    if not run.is_dir() or not run.name.startswith(manifest['lineage']):
        state['state']='waiting for the declared training lineage'
        atomic_json(status_path,state)
        return state
    candidates=[(int(p.name.removeprefix('ckpt_step_')),p) for p in complete_candidates(run)]
    if window_ready:
        checkpoint=Path(state['comparison_window']['checkpoint'])
        candidates=[(int(checkpoint.name.removeprefix('ckpt_step_')),checkpoint)]
    live_step=context['live_step']
    for task in manifest['tasks']:
        plan=plans.get(str(task.get('id'))) if isinstance(task,dict) else None
        if not plan or plan['blocked']:
            continue        # held: its status says why (plan_tasks); it never asks for a checkpoint or a window
        record=state['tasks'][task['id']]
        need=float(task['memory_gib'])
        # a mechanism that holds training when its round fails runs only inside a saved window, the pause
        # it keeps for review (reconcile_task, ability_comparison_window.advance)
        holds=task.get(HOLD_FEATURE) is True
        due=due_checkpoint(task,record,candidates,threshold=plan['run'])
        if due is None:
            # No checkpoint yet at or after the step this task may run at. Only a task whose own step to
            # pause training has come asks for one; a triggered task before it joins the next window.
            if window_ready or live_step is None or live_step<plan['open']:
                continue
            reading=commit_reading()
            fits=reading is not None and verdict(reading[0],reading[1],need,beside_training_reserve(False))
            if fits and not holds:
                control=run/'control.json'
                if not control.exists() and time.time()-state.get('snapshot_requested_at',0)>300:
                    atomic_json(control,dict(cmd='snapshot',reason='A scheduled physical ability comparison is due.'))
                    state.update(snapshot_requested_at=time.time(),snapshot_requested_step=live_step)
                    record.update(status='waiting for requested checkpoint')
            elif reading is not None and trainer_running():
                # ONE SAVE PER PAUSE (2026-09-25). A task that cannot run beside training asks for the
                # saved window at once; the window's own save is its checkpoint. Asking for a snapshot
                # first cost a second save before every pause (8:47:28 and 8:48:33 AM: about 20 s of
                # training each, and a 1.37 GB folder on C:).
                record.update(status='waiting for memory' if not fits else 'waiting for a saved window',memory_reading=reading)
                if comparison_window.request(root,folder,state,run,supervisor,live_step):
                    return state
            continue
        step,checkpoint=due
        digest,version=plan['digest'],plan['version']
        # attempts at one checkpoint count per test version too, so whatever releases a held version
        # (test_version) also gives the same checkpoint its attempts back
        attempts=int(record.get('attempts',0)) if record.get('attempt_step')==step and record.get('test_version')==version else 0
        if attempts>=2:
            record.update(status='repair needed',reason='Two recorded attempts failed; update the test and it retries automatically.')
            continue
        # NEVER IN THE RELAUNCH GAP (2026-09-24, 6:03 PM). A released window restores the
        # supervisor's run command, and the supervisor relaunches the trainer about 40 s later. A
        # memory check in that gap sees an empty machine: the 12 GB gaze test started at 6:03:39
        # PM and ran beside the relaunched trainer until 6:17 PM, the squeeze that filled C: at
        # 1:29 PM. Outside a saved window a task is judged only against a running trainer, so a
        # task too big to run beside training asks for a window instead.
        if not window_ready and not trainer_running():
            record.update(status='waiting for the supervisor to relaunch training',due_step=step)
            continue
        reading=commit_reading()
        fits=reading is not None and verdict(reading[0],reading[1],need,beside_training_reserve(window_ready))
        if not fits or (holds and not window_ready):
            record.update(status='waiting for memory' if not fits else 'waiting for a saved window',memory_reading=reading)
            if window_ready:
                if time.time()-state['comparison_window']['ready_at']>300:
                    record.update(status='memory capacity needs repair',memory_window_step=step,
                                  memory_window_script=digest,
                                  reason='The saved training stop still did not provide the declared memory after five minutes; adjust the test budget or resources.')
            elif record.get('memory_window_step')==step and record.get('memory_window_script')==digest:
                record.update(status='memory capacity needs repair',reason='A saved comparison window already failed its memory budget; adjust resources or the test before another pause.')
            elif step<plan['open'] and (live_step is None or live_step<plan['open']):
                # a triggered task before its own step to pause training joins the next saved window
                record.update(status='waiting to join the next saved window',reason='; '.join(plan['reasons']))
            # The window must cover the checkpoint this task waits for. The training statistics
            # lag the snapshot this scheduler requested: on 2026-09-24 the live step read 384000
            # while the gaze test waited for checkpoint 384025, so the window listed no task,
            # released at once, and the 4 GB test ran beside the relaunched trainer (commit 68.5
            # of 68.8 GB, "the paging file is too small"), four stop-and-restart cycles in 30 min.
            elif reading is not None:
                record['due_step']=step
                if comparison_window.request(root,folder,state,run,supervisor,max(live_step or 0,step)):
                    return state
            continue
        destination=folder/f"{task['id']}_{step}_attempt{attempts+1}_{time.time_ns()}"
        command=task_command(root,task,checkpoint,destination)
        for key in ('child','observed_children','reason','result','result_sha256','outcome','completed_at','launch_error',
                    'peak_private_commit_gib','review_hold','registry_start','registry_end'):
            record.pop(key,None)
        from .test_registry import new_run_id
        record['registry_run_id']=new_run_id(task['id'],time.time())
        record.update(status='starting',attempt_step=step,attempts=attempts+1,script_sha256=digest,
                      test_version=version,checkpoint=str(checkpoint),started_at=time.time(),
                      output=str(destination),command=command,
                      # what this round measures: the basis its triggers compare against once it completes
                      launch_basis=dict(code_commit=context['head'],flags=context['flags'],
                                        gauges=launch_gauges(record,context)),
                      hold_on_failed_comparison=(str(task.get('decided') or 'The manifest declares that a failed round '
                                                     'holds training for review.') if holds else None))
        state.update(state='running',active_task=task['id'],timestamp=time.time())
        # This write precedes spawning. Restart can find the exact unique command
        # even if this worker dies before it records the process identity.
        atomic_json(status_path,state)
        try:
            child=run_task(root,task,checkpoint,destination,command,run_id=record['registry_run_id'])
            record.update(status='running',child=child,last_seen_at=time.time())
            why=registry_started(root,task['id'],record,child)
            record['registry_start']='running' if why is None else f'not written: {why}'
        except Exception as error:
            # Popen may have succeeded before identity inspection or sidecar I/O
            # failed. Reconcile the saved command on the next cycle before retry.
            record.update(launch_error=f'{type(error).__name__}: {error}')
        state.update(timestamp=time.time())
        atomic_json(status_path,state)
        # Re-read resources and control before launching the next experiment.
        return state
    atomic_json(status_path,state)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    root=args.root.resolve()
    folder=root/'Saved/mimo_bridge/ability_rechecks'
    folder.mkdir(parents=True,exist_ok=True)
    lock=OwnerLock(folder/'owner.lock')
    try:
        while True:
            state=None
            try:state=cycle(root,args.manifest.resolve(),folder)
            except Exception as error:
                atomic_json(folder/'worker_error.json',dict(timestamp=time.time(),error=f'{type(error).__name__}: {error}'))
            if args.once:return
            time.sleep(poll_seconds(state))
    finally:lock.close()


def poll_seconds(state):
    """WINDOW_POLL_S while a saved window is being prepared, saved or used (training stands still),
    else POLL_S."""
    window=(state or {}).get('comparison_window') if isinstance(state,dict) else None
    return WINDOW_POLL_S if isinstance(window,dict) and window.get('state') in ('preparing','saving','ready') else POLL_S


if __name__=='__main__':main()
