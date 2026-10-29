# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Use learned future self functioning as an explicit, inspectable intention.

Shadow evaluation is available while learning. Acting requires a located physical
validation receipt and the exact validated self-model weights. Changing those
weights withdraws the license until evaluated again; a low predicted variance
alone never certifies that an untrained model understands its actions.
"""
import hashlib
import json
from pathlib import Path
import torch
from lem.model.homeostatic_planner import FunctionCriterion,evaluate_candidates


def fingerprint(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        digest.update(str((name,tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def perception_modules(model):
    names=('encoder_registry','forward_model','prediction_heads','motor_controller',
           'wellness','gaze_head','voice_head')
    modules={k:getattr(model,k,None) for k in names}
    modules.update(model._side_modules())
    return {k:v for k,v in modules.items() if v is not None}


def perception_fingerprint(model):
    """Bind validation to every learned dependency of perception and action."""
    digest=hashlib.sha256()
    for name,module in sorted(perception_modules(model).items()):
        digest.update(name.encode());digest.update(fingerprint(module).encode())
    return digest.hexdigest()


class SelfIntention:
    def __init__(self,mode,receipt_path):
        if mode not in ('shadow','on','qualification'):raise ValueError('invalid self intention mode')
        self.mode=mode;self.path=str(receipt_path) if not isinstance(receipt_path,dict) else 'embedded validation receipt'
        self.receipt=(json.loads(json.dumps(receipt_path,allow_nan=False)) if isinstance(receipt_path,dict)
                      else json.loads(Path(receipt_path).read_text(encoding='utf-8')))
        r=self.receipt
        self.criterion=FunctionCriterion(r['minimum_resolution'],r['maximum_uncertainty'],self.path,r.get('minimum_capacity'))
        import math
        for name in ('candidate_position_fraction','candidate_rotation_fraction'):
            if not math.isfinite(r[name]) or not 0<=r[name]<=1:raise ValueError('invalid calibrated candidate domain')
        if mode=='on':
            acquisition=(r.get('schema')=='physical_self_choice_v2' and r.get('qualification_route')=='acquisition')
            gain_fields=('acquisition_gain_lower','action_effect_gain_lower') if acquisition else ('choice_gain_lower','action_effect_gain_lower')
            for name in gain_fields:
                if not math.isfinite(r.get(name,float('nan'))) or r[name]<=0:
                    raise ValueError('physical choice gains must have finite positive lower bounds')
            if (r.get('schema') not in ('physical_self_choice_v1','physical_self_choice_v2') or r.get('body_function')!='modulate'
                    or not r.get('held_out_worlds') or not r.get('replay_verified')
                    or not r.get('retention_pass')
                    or r.get('action_effect_gain_lower',-1)<=0):
                raise ValueError('acting requires measured action effects, choice gain and retention')
            if r.get('schema')=='physical_self_choice_v2':
                if r.get('qualification_route') not in ('restoration','acquisition'):
                    raise ValueError('unknown physical qualification route')
                error=r.get('prediction_error_upper',float('nan'))
                separation=r.get('decision_effect_separation_lower',float('nan'))
                if (not math.isfinite(error) or error<0 or not math.isfinite(separation) or separation<=0
                        or 2*error>=separation or not r.get('action_prediction_scale')):
                    raise ValueError('prediction error cannot resolve the measured physical decision effect')
                if acquisition:
                    noninferiority=r.get('viability_gain_lower',float('nan'))
                    if (not math.isfinite(noninferiority) or noninferiority<0
                            or r.get('acquisition_metric') not in ('controllable_distinctions','physical_task_success')
                            or not r.get('acquisition_comparator') or not r.get('learning_opportunity')):
                        raise ValueError('healthy acquisition requires controlled gain and maintained viable functioning')
        self.last={}

    def validation_matches(self,model,frames):
        # Encoder or world-model updates can change the self predictor's inputs
        # even if its own weights are frozen. Those also require revalidation.
        versions=tuple((name,k,v.data_ptr(),v._version) for name,m in sorted(perception_modules(model).items())
                       for k,v in m.state_dict(keep_vars=True).items())
        if getattr(self,'_versions',None)!=versions:
            self._digest=fingerprint(model.self_state)
            self._perception_digest=perception_fingerprint(model)
            self._versions=versions
        r=self.receipt
        from .action_decoder import MOTOR_CONTRACT
        return (r.get('self_weights_sha256')==self._digest
                and r.get('perception_weights_sha256')==self._perception_digest
                and r.get('motor_contract')==MOTOR_CONTRACT
                and all(f.meta.get('motor_contract')==MOTOR_CONTRACT for f in frames)
                and all(f.meta.get('body_function',{}).get('profile')==r.get('body_function_profile') for f in frames)
                and all(f.meta.get('body_function',{}).get('mode')=='modulate' for f in frames))

    @torch.no_grad()
    def choose(self,model,proposal,dev,handle_state,step,frames):
        from .lem_model import command_for_encoder
        r=self.receipt
        if model.self_history.inputs is None: return None
        licensed=self.validation_matches(model,frames)
        if self.mode in ('on','qualification') and not licensed:
            self.last=dict(status='learning; physical choice license requires revalidation',executed=False)
            return None
        hold=int(r['order_steps']);horizon=int(r['orders']);dt=float(r['step_seconds'])
        if min(hold,horizon)<=0 or not 0<dt<=1 or hold*horizon>model.self_history.capacity:
            raise ValueError('unmeasured or unsupported physical prediction horizon')
        b,nh,_=proposal.shape;k=6
        raw=proposal[:,None,None].expand(b,k,horizon,nh,9).clone().float()
        # Existing proposal, achieved-pose hold, continuation, then unlabelled
        # variations. No movement such as rolling/reaching is supplied as a target.
        raw[:,1,:,:,:3]=handle_state[:,None,:,:3]
        raw[:,1,:,:,3:7]=handle_state[:,None,:,3:7]
        history=model.self_history.intention
        if history is not None and history['raw'].shape==raw[:,2].shape:
            carried=history['raw'].to(raw.device)
            previous_valid=history.get('valid',torch.ones(b,dtype=torch.bool,device=raw.device)).to(raw.device)
            raw[:,2]=torch.where(previous_valid[:,None,None,None],carried,raw[:,2])
        rng=torch.Generator().manual_seed(int(step)+93217)
        variation=torch.randn(raw[:,3:].shape,generator=rng).to(raw.device)
        raw[:,3:,:,:,:3]+=variation[...,:3]*float(r['candidate_position_fraction'])
        raw[:,3:,:,:,3:7]+=variation[...,3:7]*float(r['candidate_rotation_fraction'])
        q=raw[...,3:7];raw[...,3:7]=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        plan_state=model._capture_plan_state()
        actual=[];previous=plan_state['smooth_prev']
        previous=None if previous is None else previous[:,None].expand(b,k,nh,9).reshape(b*k,nh,9)
        hs=handle_state[:,None].expand(b,k,*handle_state.shape[1:]).reshape(b*k,*handle_state.shape[1:])
        for j in range(horizon):
            cmd=model.planned_execution(raw[:,:,j].reshape(b*k,nh,9),dev,hs,
                                        previous=previous,smoothing=plan_state['smoothing'],
                                        keep_gaze_idx=plan_state.get('smooth_keep_gaze'),
                                        eye_row_convention=plan_state.get('eye_row_convention','legacy'),
                                        executed_fields=plan_state.get('executed_fields_only',False),
                                        position_rows=plan_state.get('position_rows','all'))
            previous=cmd;actual.append(command_for_encoder(cmd).reshape(b,k,1,-1).expand(b,k,hold,-1))
        commands=torch.cat(actual,2)
        with torch.autocast(device_type=model.device.type,enabled=False):
            percept=model.self_state.perceive(**model.self_history.inputs)
            result=evaluate_candidates(model.self_state,percept,commands,
                torch.full(commands.shape[:3],dt,device=model.device),model.self_history.valid,self.criterion)
        idx=result['index'];good=idx>=0
        # One body's deficit must not assign a new intention to another body
        # whose eligible futures all remain in its own viable range.
        needs_choice=(result['cost'].masked_fill(~result['eligible'],0)>0).any(-1)
        good=good & needs_choice
        opportunity=getattr(model,'learning_opportunity_audit',None)
        opportunity_scores=None
        if opportunity is not None:
            viable=result['eligible'] & result['cost'].eq(0)
            opportunity_scores=opportunity.score(commands,viable)
            gain,preferred=opportunity_scores.max(-1)
            exploring=gain>0
            idx=torch.where(exploring,preferred,idx)
            good=good | exploring
        self.last=dict(status=('unqualified copied-world trial' if self.mode=='qualification' else
                               'validated choice' if licensed else 'shadow; not validated'),executed=False,
                       decision_step=int(step),
                       indices=idx.cpu().tolist(),costs=[[float(x) if torch.isfinite(x) else None for x in row]
                                                       for row in result['cost'].cpu()],
                       horizon_seconds=hold*horizon*dt)
        if opportunity_scores is not None:self.last['learning_opportunity']=opportunity_scores.cpu().tolist()
        if not good.any() and not needs_choice.any() and result['eligible'].any():
            model.self_history.intention=None
            self.last.update(status='predicted function stays in range; existing exploration continues',executed=False)
            return None
        if self.mode not in ('on','qualification') or not good.any():return None
        pick=raw[torch.arange(b,device=raw.device),idx.clamp_min(0)]
        selected=torch.where(good[:,None,None],pick[:,0],proposal)
        # Carry the intended continuation; the next observation can revise it.
        model.self_history.intention=dict(raw=torch.cat([pick[:,1:],pick[:,-1:]],1).detach(),
            born_step=int(step),valid=good.detach(),expected=result['mean'][torch.arange(b,device=raw.device),idx.clamp_min(0)].detach())
        self.last['executed']=True
        self.last['predicted_end_function']=model.self_history.intention['expected'][:,-1,:,0].cpu().tolist()
        return selected
