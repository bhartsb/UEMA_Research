# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Owner-facing training assessment and calculated, read-only work estimates.

No ability is inferred from elapsed time. Estimates describe recorded work only.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
import re


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def duration(seconds):
    if seconds < 90:
        return f'{max(1, round(seconds)):,} seconds'
    minutes = round(seconds / 60)
    if minutes < 60:
        return f'{minutes:,} minutes'
    hours, minutes = divmod(minutes, 60)
    return f'{hours:,} hours' + (f' {minutes} minutes' if minutes else '')


def read(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return default


def records(path):
    """Bounded start/tail reads, even after a long uninterrupted run."""
    try:
        with path.open('rb') as stream:
            first = stream.read(256_000)
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 2_000_000))
            tail = stream.read()
        rows = []
        for data in (first, tail):
            for line in data.decode('utf-8', errors='replace').splitlines():
                try:
                    row = json.loads(line)
                    if isinstance(row, dict):
                        rows.append(row)
                except ValueError:
                    pass
        # The two bounded reads can overlap.
        unique = {(r.get('type'), r.get('step'), r.get('timestamp')): r for r in rows}
        return list(unique.values())
    except OSError:
        return []


def assessment(record, args, panel):
    c = record.get('controllability') or {}
    agreement, control = c.get('agreement_mean'), c.get('control_mean')
    usable = c.get('source') == 'efference' and number(agreement) and number(control)
    encouraging = usable and agreement > control and c.get('gate_open') is True
    title = ('An encouraging prediction signal; useful movement still unproved' if encouraging else
             'Useful movement skills have not yet been demonstrated')
    evidence = ('Knowing the movement command helps predict what the body will do.' if encouraging else
                'The current measurements do not establish a repeatable movement skill.')
    comparison = None
    if usable:
        comparison = (f'Movement prediction agreement: {100*agreement:.1f}% with the actual command, '
                      f'versus {100*control:.1f}% when predicting from the same body state with another body’s command.')
    scope = ('Agreement measures both movement direction and size: 100% is an exact match, '
             '0% is no match, and negative values mean opposite movement. It is not a success rate.')
    share, orders = c.get('measured_share'), c.get('orders')
    sample = None
    if usable and number(share) and number(orders):
        sample = f'Scored {round(share*orders):,} of {int(orders):,} commands in the latest report. This is agreement in direction and size, not a success rate.'
        scope += ' Commands whose intended or predicted movement is too small to distinguish from drift are excluded.'
    on = lambda key: args.get(key) in ('on', True)
    learning, planner, exploration = on('motor_synergy'), args.get('synergy_planner', 'on') in ('on', True), on('synergy_babble')
    # whether its outputs were used is read from the record's own counts, as item 5 reads them (item 5's
    # card prints this detail); the launch settings describe the run only when the record counts no
    # executed orders by author
    from .lineage_series import synergy_outputs_used
    used = synergy_outputs_used(record, args)
    if used is not None:
        planner, exploration = used['planning'], used['exploration']
    if learning and not planner and not exploration:
        synergy = 'Learning from movement; not yet directing it'
        detail = ('Motor synergy is a network learning which body parts move together. It learns from completed '
                  'movements, but its outputs are not used to plan or explore movements.')
    elif learning:
        synergy = 'Learning coordination and issuing commands'
        uses = 'planning and exploration' if planner and exploration else 'planning' if planner else 'exploration'
        detail = f'Motor synergy learns coordinated body movements. This run also uses its outputs for {uses}.'
    else:
        synergy, detail = 'Coordination network disabled', 'This run does not train the motor-synergy network.'
    return dict(title=title, evidence=evidence, comparison=comparison, sample=sample, scale=scope if usable else None,
                worlds=(f'The {int(args["n_envs"])} bodies are independent simulations sharing one learned model, not successive frames. Each world has its own commands, contacts and history.' if number(args.get('n_envs')) else
                        'Each body is a separate simulation. The simulations share one learned model; they are not successive frames.'),
                significance='This helps toward planning useful movements. It does not show deliberate reaching or rolling, retained skills, or faster learning than the previous setup.',
                synergy=synergy, synergy_detail=detail,
                controller=('Today, the motor controller proposes joint commands. The planner searches for commands that move the body toward a remembered state it rated highly.' if on('intention') else
                            'The motor controller supplies joint commands; this run does not use the planner to search toward remembered goals.'),
                next='Test synergy planning and exploration separately, each enabled versus disabled from identical starting states. Require more commands to approach their targets, beyond measurement uncertainty, without worse movement accuracy. Repeat on new states and after more training to check retention.',
                passed=[dict(item=r['item'], name=r['name']) for r in (panel or {}).get('items', []) if r.get('beats') is True and 'item' in r and 'name' in r])


def work_timeline(root, training, record, args, now, running):
    """Estimates carry their inputs. Missing/overdue evidence never gets rolled forward."""
    result = dict(experience='Training amount is unavailable.', progress=None, events=[],
                  finish='This run continues until stopped; it has no configured finishing time.',
                  test_status='Not scheduled: no launch or completion date is recorded for the paired movement tests.',
                  test_duration='Runtime is not measured for this paired test, so no completion estimate is available.',
                  ability_timing='No measured learning curve tells us when useful reaching or rolling should appear.')
    if not training.get('run'):
        return result
    run = (root / training['run']).resolve()
    if not run.is_relative_to(root.resolve()):
        return result
    rows = records(run / 'replay_training_stats.jsonl')
    launch = next((r for r in rows if r.get('type') == 'launch'), {})
    start, step = launch.get('step'), record.get('step')
    stamp = record.get('timestamp')
    measurements = sorted((r for r in rows if r.get('type') == 'mimo' and number(r.get('step')) and number(r.get('timestamp'))), key=lambda r:r['timestamp'])[-11:]
    speed, span = None, None
    if len(measurements) >= 2:
        span = measurements[-1]['timestamp'] - measurements[0]['timestamp']
        delta = measurements[-1]['step'] - measurements[0]['step']
        if span > 0 and delta > 0:
            speed = delta/span
    n = args.get('n_envs')
    if number(start) and number(step) and step >= start:
        result['inherited_updates'] = int(start)
        result['experience'] = (f'{int(step-start):,} new training updates since this restart, added to {int(start):,} inherited updates. '
                                f'One model learns from {n} independent simulated worlds.' if number(n) else
                                f'{int(step-start):,} new training updates since this restart, added to {int(start):,} inherited updates.')
        if number(launch.get('timestamp')) and number(stamp):
            result['progress'] = f'Those new updates took {duration(max(0, stamp-launch["timestamp"]))} of computer time.'
    fresh = running and number(stamp) and 0 <= now-stamp < 600
    def event(title, due, detail, calculation, kind='automatic'):
        result['events'].append(dict(title=title, due=due if fresh else None, detail=detail,
            calculation=calculation, kind=kind,
            status='Overdue — completion not recorded' if fresh and number(due) and due < now else
                   'Estimated' if fresh and number(due) else 'Estimate unavailable while training is paused or unconfirmed'))
    if speed and span:
        result['rate'] = dict(updates_per_second=speed, window_seconds=span, timestamp=stamp)
        cadence = args.get('report_every')
        if number(cadence) and cadence > 0 and number(step):
            remaining = cadence - step % cadence
            seconds = remaining/speed
            event('Next training report', stamp+seconds,
                  'Updates prediction measurements and body poses. It does not test repeatable skills.',
                  f'{remaining:,.0f} updates ÷ {speed:.2f} updates/second = about {duration(seconds)} after the last report. Speed measured across the last {duration(span)}.')
        result['practice_estimate'] = f'Another 1,000 training updates would take about {duration(1000/speed)} at the {"current" if fresh else "last recorded"} measured speed.'
        max_steps, minutes = args.get('max_steps', 0), args.get('minutes', 0)
        if number(max_steps) and max_steps > 0 and number(step):
            remaining = max_steps - step
            if remaining is not None:
                result['finish'] = f'Configured training limit: global update {int(max_steps):,}; about {duration(max(0, remaining)/speed)} remain at the measured speed.'
        elif number(minutes) and minutes > 0 and number(launch.get('timestamp')):
            result['finish'] = f'Configured time limit: about {duration(max(0, launch["timestamp"]+minutes*60-now))} remain.'
    saves = read(run/'checkpoints.json', [])
    if not isinstance(saves, list):
        saves = []
    hourly = sorted((s for s in saves if isinstance(s, dict) and s.get('reason') == 'hourly' and number(s.get('elapsed_min'))), key=lambda s:s['elapsed_min'])
    period = args.get('checkpoint_every_min')
    if hourly and number(period) and period > 0 and not record.get('checkpoint_failures'):
        try:
            last = datetime.strptime(hourly[-1]['time'], '%Y-%m-%d %H:%M:%S').timestamp()
            extra = None
            if len(hourly) >= 2:
                extra = max(0, (hourly[-1]['elapsed_min']-hourly[-2]['elapsed_min']-period)*60)
            delay = period*60 + (extra or 0)
            event('Next saved model', last+delay,
                  'Preserves the model and world states. Tests must be run separately.',
                  f'Last hourly save + {period:g} minutes' + (f' + about {duration(extra)} for saving and loop overhead, measured from the previous hourly interval.' if extra is not None else '. Save duration has not yet been measured; the estimate is for starting the save.'))
        except (KeyError, ValueError, TypeError, OSError):
            pass
    log = run/'hourly_watch.log'
    log_tail = ''
    if log.is_file():
        with log.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-100_000))
            log_tail = stream.read().decode('utf-8', errors='replace')
        # A watcher startup is not a completed health review.
        log_tail = log_tail[log_tail.rfind('Monitoring ') :] if 'Monitoring ' in log_tail else log_tail
    if re.search(r'H84 check-\d+\s+[\d.]+ hours into the run', log_tail):
        checked = log.stat().st_mtime
        event('Next health review (approximate)', checked+3600,
              'Checks training errors, software responsiveness and memory; it can stop training for a declared fault. It does not test skills.',
              'Last health-report write + the watcher’s one-hour interval. Runtime and process interruptions can delay it.')
    result['events'].sort(key=lambda e: e['due'] if number(e['due']) else float('inf'))
    result['source'] = '/files/'+(run/'checkpoints.json').relative_to(root).as_posix()
    return result
