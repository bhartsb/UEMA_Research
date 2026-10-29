# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Token-owned saved pause for scheduled comparisons; the supervisor alone resumes.

The worker never kills or launches a trainer. Any operator control change cancels
its authority to resume. A damaged or missing final save leaves a visible stop.
"""
from pathlib import Path
import json
import os
import time
import uuid

from .training_continuation import atomic_json, complete_candidates, pause_reason, processes
from ..storage_locations import locate, long_path, sha256_file


def read(path, default=None):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:return default


# an unreadable supervisor control compares unequal to every control a window owns
UNREADABLE={'unreadable_supervisor_control':True}


def read_control(path):
    """supervisor_control.json as a person may write it: UTF-8, UTF-8 with a byte-order mark, or UTF-16
    (PowerShell 5.1's '>' and Out-File), decoded as the trainer decodes its own control file. None when
    the file is absent, UNREADABLE when it cannot be decoded or parsed. A UTF-16 control written to
    release a review hold raised in every cycle before 2026-09-25 (the review of the scheduler's classes).
    The supervisor (training_continuation.pause_reason) reads UTF-8 only: it keeps training paused as
    'unreadable supervisor control' until the file is rewritten, and the scheduler reports that pause."""
    try:
        raw=Path(path).read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return UNREADABLE
    try:
        text=raw.decode('utf-16') if raw[:2] in (b'\xff\xfe',b'\xfe\xff') else raw.decode('utf-8-sig')
        value=json.loads(text)
    except (UnicodeError,ValueError):
        return UNREADABLE
    return value if isinstance(value,dict) else UNREADABLE


def save(folder,state):
    state['timestamp']=time.time()
    atomic_json(folder/'status.json',state)


def first_eligible(record,minimum_step):
    """The first step a task may run at in a window: its due step, or the earlier step from which a
    triggered task joins a saved window (ability_rechecks.plan_tasks 'ride_step')."""
    due=int(record.get('due_step',minimum_step+1))
    ride=record.get('ride_step')
    return min(due,int(ride)) if isinstance(ride,int) and not isinstance(ride,bool) else due


def request(root,folder,state,run,supervisor,minimum_step):
    """Persist ownership before issuing either pause or snapshot request."""
    if state.get('comparison_window',{}).get('state') in ('preparing','saving','ready','held for review'):
        return False
    saved=root/'Saved/mimo_bridge'
    if pause_reason(saved,run) or (run/'control.json').exists():return False
    if time.time()-float(supervisor.get('timestamp',0))>180:return False
    trainers=processes('trainer',run)
    if len(trainers)!=1 or len(processes('trainer'))!=1:return False
    previous=read_control(saved/'supervisor_control.json')
    if previous is UNREADABLE or (previous is not None and previous.get('cmd')!='run'):return False
    token=uuid.uuid4().hex
    control=dict(previous or {},cmd='pause',ability_window_token=token,
                 reason='Saved comparison window: scheduled ability tests need training memory.')
    window=dict(state='preparing',token=token,run=str(run),started_at=time.time(),
                minimum_step=int(minimum_step),previous_control=previous,owned_control=control,
                trainer_pid=trainers[0].pid,trainer_created_at=trainers[0].create_time(),
                targets={k:first_eligible(r,minimum_step) for k,r in state['tasks'].items()
                         if first_eligible(r,minimum_step)<=minimum_step})
    state['comparison_window']=window
    save(folder,state)
    advance(root,folder,state,run)
    return True


def release(root,folder,state,reason):
    """Restore only our exact control token; never override a repair/operator stop."""
    window=state['comparison_window'];saved=root/'Saved/mimo_bridge';run=Path(window['run'])
    control=read_control(saved/'supervisor_control.json')
    # This deliberately checks the run's stop records without our supervisor pause.
    repair=pause_reason(folder/'no_supervisor_control',run)
    supervisor=read(saved/'training_continuation.json',{})
    if control!=window['owned_control'] or Path(supervisor.get('active_run','')).resolve()!=run.resolve():
        window.update(state='ownership changed',reason='Control or training owner changed; automatic resume withheld.')
    elif repair:
        window.update(state='repair required',reason=repair)
    else:
        restored=window['previous_control'] or {'cmd':'run'}
        atomic_json(saved/'supervisor_control.json',restored)
        window.update(state='released',reason=reason,released_at=time.time())
    atomic_json(folder/('window_'+window['token']+'.json'),window)
    save(folder,state)


def open_reviews(state):
    """Tasks whose failed round awaits the operator's review (ability_rechecks reconcile_task)."""
    return {k:r['review_hold'] for k,r in (state.get('tasks') or {}).items()
            if isinstance(r,dict) and isinstance(r.get('review_hold'),dict) and not r['review_hold'].get('released_at')}


def release_instruction(window):
    """How to end a review hold, naming the keys the pause's control must keep (review of 2026-09-25:
    '{"cmd": "run"}' alone dropped any other key, for example "archiver": "off")."""
    previous=window.get('previous_control') if isinstance(window.get('previous_control'),dict) else {}
    kept={k:v for k,v in previous.items() if k not in ('cmd','reason','timestamp','ability_window_token')}
    keep=('and keep ' + ', '.join(f'"{k}": {json.dumps(v)}' for k,v in kept.items()) + ' as it was before this pause'
          if kept else 'and keep every other key')
    return ('To resume training after the review, edit only the "cmd" field of Saved/mimo_bridge/supervisor_control.json '
            f'to "run" {keep}; write the file as UTF-8 (Git Bash echo or an editor): PowerShell > writes UTF-16, which '
            'the supervisor cannot read, so training would stay paused.')


def hold_for_review(folder,state,reviews):
    """Keep the saved pause instead of releasing it: the supervisor control stays this window's pause.

    A MECHANISM THAT STEERS PRODUCTION HOLDS TRAINING WHEN ITS ROUND FAILS (the operator, 2026-09-25
    9:09 AM PDT: "rather lose some wall-clock time than accumulate hours of misleading experience").
    H84 runs hand goals from memory, so a failed round of the remembered hand-goal comparison keeps the
    pause for review. The hold ends only when someone changes supervisor_control.json (held_by_review):
    a control change inside a window cancels its authority, as everywhere in this module."""
    window=state['comparison_window']
    window.update(state='held for review',held_at=time.time(),review=reviews,
                  reason=(' '.join(str(r.get('reason') or '') for r in reviews.values()).strip()+' '
                          +release_instruction(window)))
    atomic_json(folder/('window_'+window['token']+'.json'),window)
    save(folder,state)


def held_by_review(root,folder,state,run):
    """While held for review: False, and nothing changes, until the supervisor control is no longer
    this window's pause (an unreadable file included) or the supervisor holds another run; then the hold
    is recorded as released."""
    window=state['comparison_window'];saved=root/'Saved/mimo_bridge'
    control=read_control(saved/'supervisor_control.json')
    supervisor=read(saved/'training_continuation.json',{})
    same_run=(Path(supervisor.get('active_run','')).resolve()==Path(window['run']).resolve()
              and Path(run).resolve()==Path(window['run']).resolve())
    if control==window['owned_control'] and same_run:
        return False
    now=time.time()
    # the failed round stays in window['review'] and in the task's review_hold
    window.update(state='review released',review_released_at=now,
                  reason=('The supervisor control or its run changed during the review hold, so this window no longer '
                          'holds training; the supervisor resumes it when the control says run.'
                          +(' The control file could not be read, so the supervisor keeps training paused until it is '
                            'rewritten as UTF-8.' if control is UNREADABLE else '')))
    for key in window.get('review') or {}:
        record=(state.get('tasks') or {}).get(key)
        if isinstance(record,dict) and isinstance(record.get('review_hold'),dict):
            record['review_hold'].setdefault('released_at',now)
    atomic_json(folder/('window_'+window['token']+'.json'),window)
    save(folder,state)
    return False


def advance(root,folder,state,run,runnable=None):
    """Return True only while a verified saved window permits a new comparison.

    runnable(step): the tasks that can still run on the window's checkpoint this cycle (the scheduler's
    own plan: due there, not held, not already done there, attempts left). A PAUSE ALWAYS ENDS OR IS
    REPORTED (review of 2026-09-25): the list of tests fixed at the request kept a pause open, training
    stopped, when a listed test could no longer run on its checkpoint (a trigger that stopped firing, an
    old review hold). With runnable, the window waits exactly for what can run, a test that became due
    on the checkpoint runs in it instead of opening a second pause, and the listed tests that did not
    run are named in the window's record. Without it, the listed tests (targets) decide, as before."""
    window=state.get('comparison_window') or {}
    if window.get('state')=='held for review':
        return held_by_review(root,folder,state,run)
    if window.get('state') not in ('preparing','saving','ready'):return False
    saved=root/'Saved/mimo_bridge';control_path=saved/'supervisor_control.json'
    if Path(window['run']).resolve()!=run.resolve():
        window.update(state='ownership changed',reason='Supervisor moved to another run; no control was changed.')
        save(folder,state);return False
    control=read_control(control_path)
    if window['state']=='preparing' and control==window['previous_control']:
        # Restart recovery for a crash before our first control write.
        if pause_reason(saved,run):
            window.update(state='repair required',reason='A stop superseded the scheduled comparison.')
            save(folder,state);return False
        atomic_json(control_path,window['owned_control'])
        control=window['owned_control']
    if control!=window['owned_control']:
        window.update(state='ownership changed',reason='Operator control changed; no automatic resume is authorized.')
        save(folder,state);return False
    repair=pause_reason(folder/'no_supervisor_control',run)
    if repair:
        window.update(state='repair required',reason=repair)
        save(folder,state);return False
    if window['state']=='preparing':
        request_path=run/'control.json'
        request=dict(cmd='snapshot_and_stop',auto_resume=True,ability_window_token=window['token'],
                     reason='Preserve all eight worlds for scheduled physical comparisons.')
        existing=read(request_path)
        consumed=any(read(path,{}).get('ability_window_token')==window['token'] for path in run.glob('control_done_*.json'))
        if existing is None and not consumed:
            try:
                with request_path.open('x',encoding='utf-8') as stream:json.dump(request,stream)
            except FileExistsError:existing=read(request_path)
        if existing is not None and existing!=request:
            window.update(state='control conflict',reason='Another trainer control arrived; automatic resume withheld.')
            save(folder,state);return False
        window.update(state='saving',save_requested_at=time.time())
        save(folder,state)
    if window['state']=='saving':
        if processes('trainer',run):
            state['state']='saving training before scheduled comparisons'
            if time.time()-window['save_requested_at']>1200:
                window.update(state='repair required',reason='The trainer did not finish its saved stop within 20 minutes; inspect it.')
            save(folder,state);return False
        done=read(run/'stage_done.json',{})
        candidates=complete_candidates(run)
        checkpoint=next(iter(candidates),None)
        episode=read(checkpoint/'episode_state.json',{}) if checkpoint else {}
        consumed=[read(path,{}) for path in run.glob('control_done_*.json')]
        our_stop=any(c.get('ability_window_token')==window['token'] and c.get('cmd')=='snapshot_and_stop'
                     and c.get('auto_resume') is True for c in consumed)
        try:
            physical=locate(episode.get('storage_path',''),expected_sha256=episode.get('sha256'))
            # the companion's path repeats the run name and passes Windows' 260 characters for a
            # supervisor resume folder (262 on 2026-09-24, "..._seg38_observer_resume_<time>");
            # Path.is_file() then says False for a file that is there, so the checks go through
            # the extended-length form, as the writer and continuation_state.load do
            physical_verified=(os.path.isfile(long_path(physical)) and os.stat(long_path(physical)).st_size==episode.get('bytes')
                               and bool(episode.get('sha256')) and sha256_file(physical)==episode['sha256'])
            final_checkpoint=locate(done.get('final_checkpoint',''))
        except (OSError,ValueError):
            physical_verified=False;final_checkpoint=Path('')
        if (not checkpoint or done.get('ok') is not True or done.get('finished_at',0)<window['started_at']
                or not our_stop or final_checkpoint.resolve()!=checkpoint.resolve()
                or int(done.get('steps',-1))<window['minimum_step']
                or episode.get('step')!=done.get('steps') or len(episode.get('times',[]))!=8
                or not physical_verified):
            window.update(state='repair required',reason='Final complete model and eight-world save could not be verified; training stays paused.')
            save(folder,state);return False
        window.update(state='ready',checkpoint=str(checkpoint),episode_sha256=episode['sha256'],
                      episode_storage=str(physical),ready_at=time.time())
    if window['state']=='ready':
        if processes('trainer'):
            window.update(state='ownership changed',reason='Another trainer started; new comparisons are withheld.')
            save(folder,state);return False
        if runnable is not None:
            step=int(Path(window['checkpoint']).name.removeprefix('ckpt_step_'))
            pending=sorted(runnable(step))
            # the listed tests that will not run on this checkpoint, and why (the scheduler's status says)
            window['not_run']={key:str((state['tasks'].get(key) or {}).get('status') or 'not due on this checkpoint')
                               for key in window['targets'] if key not in pending
                               and int((state['tasks'].get(key) or {}).get('completed_step',-1))<step}
        else:
            pending=[key for key,step in window['targets'].items()
                     if state['tasks'][key].get('completed_step',-1)<step
                     and state['tasks'][key].get('status') not in ('repair needed','memory capacity needs repair',
                                                                   'held for review')]
        if not pending:
            reviews=open_reviews(state)
            if reviews:
                hold_for_review(folder,state,reviews)
                return False
            release(root,folder,state,'Scheduled comparisons recorded; the existing supervisor resumes the saved run.')
            return False
        state['state']='saved comparison window'
        return True
    return False
