# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Ground the additional self-state in each body's actual experienced observations."""
import logging
import torch
import torch.nn.functional as F
from lem.body_function import CONTRACT,FUNCTION_FIELDS
from lem.model import self_state as self_state_module
from lem.model.self_state import SITE_INPUT, MOTOR_GROUPS, map_state
from .skin_contract import frame_skin_features

log=logging.getLogger(__name__)


def batch_observation(frames,device):
    n=max(len((f.meta or {}).get('body_function',{}).get('values',[])) for f in frames)
    if not n:raise ValueError('self perception requires the body-function producer')
    sites=torch.zeros(len(frames),n,SITE_INPUT,device=device)
    valid=torch.zeros(len(frames),n,device=device,dtype=torch.bool)
    for i,f in enumerate(frames):
        record=(f.meta or {}).get('body_function')
        if not record or record.get('contract')!=CONTRACT:
            raise ValueError('body-function contract missing or incompatible')
        if record.get('mode') not in ('observe','modulate'):
            raise ValueError('unknown body observation regime')
        if tuple(record.get('fields',()))!=FUNCTION_FIELDS:
            raise ValueError('body-function field meanings do not match the contract')
        q=torch.as_tensor(record['values'],device=device,dtype=torch.float32)
        if record.get('time') != f.sim_time:raise ValueError('body-function observation is stale')
        if q.ndim!=2 or q.shape[1]!=3 or len(record['site_ids'])!=len(q):
            raise ValueError('invalid body-function site layout')
        if not torch.isfinite(q).all() or ((q<0)|(q>1)).any():
            raise ValueError('body-function values outside the stated physical range')
        touch=frame_skin_features(f).to(device)
        pos=getattr(f,'touch_pos',None)
        if pos is None or len(pos)!=len(q) or len(touch)<len(q):
            raise ValueError('self perception cannot invent skin location or tactile data')
        sites[i,:len(q)]=torch.cat([torch.as_tensor(pos,device=device),touch[:len(q),:8],q],-1)
        valid[i,:len(q)]=True
    if not torch.isfinite(sites).all():raise ValueError('nonfinite local self state')
    now=torch.tensor([f.sim_time for f in frames],device=device,dtype=torch.float64)
    return sites,valid,now


WINDOW_FIELDS=('steps','live_over_limit','releases','body_releases','restored_bodies','bounded_calls',
               'belief_size_max','attention_peak_max','attention_capped_steps','cleared_rehearsal_windows',
               'cleared_events')


def new_window():
    """Per-step counts for one stats report (docs/gain_of_function_v4/agents/self_percept_runaway_2026-09-26.md).

    A stats record carries the self-state of its LAST step only, so a percept that ran over the limit
    on any other step of the report never showed (review d932dcbcd: a body stuck again every 144 steps
    read under 8 in 10 of 10 records). These count every step since the previous record:
      steps                     steps observed in this window
      live_over_limit           steps whose live percept was above the limit (a rescale with the switch
                                off, a release with it on)
      releases, body_releases   steps with a release, and bodies released
      restored_bodies           bodies released before the live call because their stored percepts sat
                                at the limit (memory written without the switch; the first step of a
                                resume)
      bounded_calls             perceive() calls above the limit, the live call and the recomputed ones
                                (forecast lessons, rehearsal) - the old belief_bounded_count's growth
      belief_size_max           the largest live percept size on any step (belief_size is the last step's)
      attention_peak_max        the largest share of a body's recall one stored percept carried, before
                                the cap (switch on only; healthy 0.009 to 0.033, stuck 1.000)
      attention_capped_steps    steps on which the cap changed the live recall of at least one body
      cleared_rehearsal_windows, cleared_events   saved inputs whose stored percepts a release cleared
    """
    return dict(steps=0,live_over_limit=0,releases=0,body_releases=0,restored_bodies=0,bounded_calls=0,
                belief_size_max=None,attention_peak_max=None,attention_capped_steps=0,
                cleared_rehearsal_windows=0,cleared_events=0)


def close_window(model):
    """Start the next report's counts; train_mimo.flush_stats calls this after copying the record."""
    module=getattr(model,'self_state',None)
    if module is not None:module.percept_window=new_window()


def held_bodies(relations):
    """[bodies, rows, width] -> [bodies] True where a stored percept sits at the size limit.

    Only the old storing path writes a percept at the limit (it rescales to exactly 8); with the
    switch on, a percept over the limit is stored as zero. So a held body is memory carried from a
    run without the switch."""
    return (relations.abs().amax(-1)>=self_state_module.BELIEF_LIMIT*(1-1e-5)).any(1)


def zero_bodies(relations,bodies):
    return torch.where(bodies.to(relations.device)[:,None,None],torch.zeros_like(relations),relations)


def clear_released(model,bodies):
    """The released bodies' stored percepts leave the saved inputs that would bring them back.

    history.events and history.inputs hold the stored percepts of up to 128 earlier steps; the
    forecast lessons recompute past percepts from them, and rehearsal windows are copied from them
    every 20 steps, so each keeps the old memory alive after the release. Every one of the released
    bodies' rows there is set to zero, as in the memory itself. A retained rehearsal window is
    cleared only where the released body holds a percept at the limit (3 of 128 windows at 415,075;
    the others are healthy lessons). Returns (events cleared, windows cleared)."""
    history=model.self_history;events=0;windows=0
    seen=set()
    for inputs in [e.get('inputs') for e in history.events]+[history.inputs]:
        if not inputs or id(inputs) in seen:continue
        seen.add(id(inputs))
        rel=inputs.get('relations')
        if rel is None:continue
        mask=bodies.to(rel.device)
        if bool((rel[mask]!=0).any()):
            inputs['relations']=zero_bodies(rel,mask);events+=1
    rehearsal=getattr(model,'self_rehearsal',None)
    if rehearsal is not None:
        for entry in list(rehearsal.common)+[e for _,e in rehearsal.changed]:
            rel=entry['inputs'].get('relations')
            if rel is None:continue
            held=held_bodies(rel)&bodies.to(rel.device)
            if bool(held.any()):
                entry['inputs']['relations']=zero_bodies(rel,held);windows+=1
    return events,windows


def observe(model,tokens,frames,executed,spatial=None,spatial_valid=None):
    from .lem_model import command_for_encoder
    module=model.self_state;history=model.self_history
    release_on=bool(getattr(model,'self_percept_release',False))
    window=getattr(module,'percept_window',None)
    if window is None:window=module.percept_window=new_window()
    bounded_before=int(getattr(module,'belief_bounded_count',0))
    restored=None
    if release_on and history.relations:
        # --self-percept-release on, before the live call: a body whose stored percepts sit at the
        # limit (memory carried from a run without the switch) starts again from zero now, so the
        # first live percept of the resume is not recalled from the stuck memory.
        held=held_bodies(torch.stack(list(history.relations),1))
        if bool(held.any()):
            restored=held
            history.release(held)
            events,windows=clear_released(model,held)
            window['restored_bodies']+=int(held.sum())
            window['cleared_events']+=events;window['cleared_rehearsal_windows']+=windows
            log.warning('self perception: bodies %s held stored percepts at the limit of %.4g (memory '
                        'written without --self-percept-release); their stored percepts were set to zero '
                        'before this step, with %d saved step inputs and %d rehearsal windows',
                        [int(i) for i in held.nonzero().flatten()],self_state_module.BELIEF_LIMIT,events,windows)
    sites,valid,now=batch_observation(frames,model.device)
    identities=tuple(tuple(f.meta['body_function']['site_ids']) for f in frames)
    if any(len(set(ids))!=len(ids) for ids in identities):
        raise ValueError('duplicate body-function site identity')
    if history.site_ids is not None and identities!=history.site_ids:
        raise ValueError('body-function site identity changed without reset')
    context=torch.cat([tokens[m].detach().float().reshape(len(frames),-1) for m in model.active],-1)
    capacity=torch.zeros(len(frames),MOTOR_GROUPS,device=model.device)
    capacity_valid=torch.zeros_like(capacity,dtype=torch.bool)
    for bi,f in enumerate(frames):
        cap=(f.meta or {}).get('motor_function')
        if cap is not None:
            if cap.get('contract')!='motor_function_v1':raise ValueError('unknown motor function contract')
            capacity[bi]=torch.as_tensor(cap['values'],device=model.device)
            capacity_valid[bi]=torch.as_tensor(cap['valid'],device=model.device,dtype=torch.bool)
    observed=torch.cat([sites[...,-3:],torch.stack([capacity,torch.zeros_like(capacity),torch.zeros_like(capacity)],-1)],1)
    if history.time is not None:
        if now.shape!=history.time.shape or (now<=history.time).any() or sites.shape!=history.sites.shape:
            raise ValueError('self-state history crossed a body/time boundary without reset')
    delta=None if history.time is None else (now-history.time).float()
    relations=None if not history.relations else torch.stack(list(history.relations),1)
    with torch.autocast(device_type=model.device.type,enabled=False):
        inputs=dict(sites=sites,valid=valid,context=context,previous=history.state,
                    internal=history.internal,dt=delta,relations=relations,
                    capacity=capacity,capacity_valid=capacity_valid,
                    regime=torch.tensor([f.meta['body_function']['mode']=='modulate' for f in frames],device=model.device).long())
        if spatial is not None:inputs.update(spatial=spatial,spatial_valid=spatial_valid)
        extent=[f.meta.get('skin_function_granularity') for f in frames]
        if module.prediction_mode=='grounded_residual' and any(e is not None for e in extent):
            if not all(e is not None for e in extent):raise ValueError('mixed skin granularity lacks valid local extent')
            inputs['function_extent']=torch.tensor([
                list(zip(e['area_weighted_quality'],e['affected_sensor_area_fraction'])) for e in extent],device=model.device)
            if any('surface_moments_root' in e for e in extent):
                if not all('surface_moments_root' in e for e in extent):raise ValueError('incomplete spatial skin observations')
                inputs['surface_moments']=torch.tensor([e['surface_moments_root'] for e in extent],device=model.device)
        percept=module.perceive(**inputs)
        # read before any other perceive() call below overwrites it: the horizon and
        # rehearsal lessons recompute past percepts, and only this one is stored
        live_size=getattr(module,'last_belief_size',None)
        live_peak=getattr(module,'last_attention_peak',None) if module.attention_peak_limit is not None else None
        live_capped=int(getattr(module,'last_attention_capped',0))
        full_valid=percept['valid']
        mask=full_valid[:,:,None].expand(-1,-1,3)
        present_loss=module.present_loss(percept,observed)
        loss=present_loss
        learned_present=percept.get('learned_present',percept['present'])
        stats={'present_error':float((learned_present-observed).square()[mask].mean().detach()),
               'present_training_loss':float(present_loss.detach()),'valid_sites':int(valid.sum()),
               'resolution_min':float(sites[...,-3].masked_fill(~valid,1).min()),
               'motor_capacity_min':float(capacity.masked_fill(~capacity_valid,1).min()) if capacity_valid.any() else None,
               'consequence_error':None,'relation_events':len(history.relations),
               # the stored percept's size before any rescale, largest body: healthy on H84
               # 1.4 to 2.0 at seven checkpoints from step 344,400 to 405,325; the limit is
               # 8 (self_state.BELIEF_LIMIT); 16 to 25 while one body was latched at it
               'belief_size':None if live_size is None else float(live_size.max())}
        if module.prediction_mode == 'grounded_residual':
            stats['prediction_mode']=module.prediction_mode
            stats['present_source']='observed local function; learned belief retained separately'
            readout_error=(percept['learned_present']-observed).abs()
            stats['learned_present_worst_site_error']=float(readout_error.masked_fill(~mask,0).max().detach())
            stats['learned_present_function_worst_error']=float(readout_error[...,0].masked_fill(~full_valid,0).max().detach())
            # A plumbing check, not a reading. In this mode percept['present'] IS the
            # measurement (self_state.py:105), so the line below is a tensor minus
            # itself: 0.0 for every possible input, forever. It confirms the published
            # present was not silently replaced by a belief, and it must never be shown
            # as how well the model knows its own body. That reading is
            # learned_present_worst_site_error above, which is what the explainer's
            # 'belief vs measured now' row displays.
            stats['observed_grounding_error']=float((percept['present']-observed).abs()[mask].max().detach())
            if stats['observed_grounding_error']:
                raise ValueError('published present is not the measured body function')
            stats['spatial_regions']=0 if spatial_valid is None else int(spatial_valid.sum())
            # times the self-percept had to be rescaled to stay inside its limit; 0 on a
            # healthy run, and any value above 0 names the runaway of 2026-09-20
            stats['belief_bounded_count']=int(getattr(module,'belief_bounded_count',0))
            stats['function_extent_observed']='function_extent' in inputs
            stats['surface_moments_observed']='surface_moments' in inputs
        if history.sites is not None and executed is not None:
            # Recompute the previous perception with current weights, using only
            # information available then. The current outcome is the target only.
            action=command_for_encoder(executed.detach().float()).reshape(len(frames),1,-1)
            history.events.append(dict(inputs=history.inputs,action=action[:,0],dt=delta,
                initial_target=history.targets,target=observed.detach(),valid=history.valid & full_valid))
            if module.prediction_mode=='grounded_residual':
                # About 1.0 when the stored command calibration matches the orders the
                # body is issuing; climbs as it goes stale (H83 checkpoint: 6.4159). P203.
                stats['command_normalised_absmean']=float(
                    ((action[:,0]-module.grounded.action_center)/module.grounded.action_scale).abs().mean())
            # Supervise full executed sequences, not an invented command held for
            # the entire interval. Endpoints and every intervening outcome are real.
            losses=[]
            horizons=(1,5,20)+( (100,) if model.step_count%20==0 else () )
            for horizon in horizons:
                if len(history.events)<horizon:continue
                events=list(history.events)[-horizon:]
                previous=module.perceive(**events[0]['inputs'])
                actions=torch.stack([e['action'] for e in events],1)
                durations=torch.stack([e['dt'] for e in events],1)
                mean,var=module.forecast(previous,actions,durations)
                target=torch.stack([e['target'] for e in events],1)
                future_valid=torch.stack([e['valid'] for e in events],1)
                losses.append(module.learning_loss(mean,var,target,future_valid,initial=events[0].get('initial_target')))
                err=((mean-target).square()*future_valid[...,None]).sum()/(3*future_valid.sum()).clamp_min(1)
                stats['consequence_error_'+str(horizon)]=float(err.detach())
                # The floor this forecast has to beat, on the same window. The change
                # head's zero output IS persistence, so holding the starting
                # measurement for the whole window is the do-nothing control. Without
                # it a forecast error carries no reference and reads as near-perfect
                # at any value.
                still=previous.get('observed')
                if still is not None:
                    base=((still[:,None].expand_as(mean)-target).square()*future_valid[...,None]).sum()/(3*future_valid.sum()).clamp_min(1)
                    stats['persistence_error_'+str(horizon)]=float(base.detach())
                    stats['forecast_vs_persistence_'+str(horizon)]=(float((err/base).detach())
                                                                    if float(base)>0 else None)
                # What the module says its own spread is, beside the error above. The
                # two track each other only if that stated uncertainty is calibrated.
                stats['forecast_stated_std_'+str(horizon)]=float(var.sqrt().mean().detach())
            loss=loss+torch.stack(losses).mean()
            stats['consequence_error']=stats['consequence_error_1']
            if len(history.events)>=20 and model.step_count%20==0:
                length=100 if len(history.events)>=100 and model.step_count%100==0 else 20
                model.self_rehearsal.add(list(history.events)[-length:])
            if model.step_count%5==0:
                lesson=model.self_rehearsal.sample(model.device)
                if lesson is not None:
                    recalled=module.perceive(**lesson['inputs'])
                    mean,var=module.forecast(recalled,torch.stack(lesson['actions'],1),torch.stack(lesson['dt'],1))
                    recall_loss=module.learning_loss(mean,var,torch.stack(lesson['targets'],1),torch.stack(lesson['valid'],1),
                                                    initial=recalled.get('observed'))
                    loss=loss+recall_loss
                    stats['rehearsal_loss']=float(recall_loss.detach())
            stats['retained_windows']=len(model.self_rehearsal.common)+len(model.self_rehearsal.changed)
        history.state=percept['state'].detach()
        history.time=now.detach();history.sites=sites.detach();history.valid=full_valid.detach();history.context=context.detach()
        history.targets=observed.detach()
        # --self-percept-release (off by default): a body whose live percept had to be
        # rescaled starts its stored percepts again from zero (SelfHistory.store says why)
        released=None
        if release_on and live_size is not None:
            over=live_size>self_state_module.BELIEF_LIMIT
            if bool(over.any()):released=over
        cleared=history.store(percept['belief'].detach(),released)
        history.inputs=map_state(inputs,model.device)
        if released is not None:
            module.self_percept_releases=getattr(module,'self_percept_releases',0)+1
            window['releases']+=1;window['body_releases']+=int(released.sum())
            events,windows=clear_released(model,released)
            window['cleared_events']+=events;window['cleared_rehearsal_windows']+=windows
            count=module.self_percept_releases
            if count<=10 or count%100==0:
                log.warning('self perception: release %d - bodies %s reached %s, above the limit of '
                            '%.4g; their %d stored percepts were set to zero and start again, with %d '
                            'saved step inputs and %d rehearsal windows',
                            count,[int(i) for i in released.nonzero().flatten()],
                            ['%.4g'%float(v) for v in live_size[released]],
                            self_state_module.BELIEF_LIMIT,cleared,events,windows)
        stats['self_percept_releases']=int(getattr(module,'self_percept_releases',0))
        history.site_ids=identities
        # every step, not only the report's last one (new_window says why)
        window['steps']+=1
        if live_size is not None:
            size=float(live_size.max())
            window['live_over_limit']+=int(size>self_state_module.BELIEF_LIMIT)
            window['belief_size_max']=size if window['belief_size_max'] is None else max(window['belief_size_max'],size)
        if live_peak is not None:
            peak=float(live_peak.max())
            window['attention_peak_max']=peak if window['attention_peak_max'] is None else max(window['attention_peak_max'],peak)
        window['attention_capped_steps']+=int(live_capped>0)
        window['bounded_calls']+=int(getattr(module,'belief_bounded_count',0))-bounded_before
        stats.update({'window_'+k:window[k] for k in WINDOW_FIELDS})
    model._self_percept={k:v.detach() for k,v in percept.items()}
    model.last_self_state=stats
    return loss,percept['bias']
