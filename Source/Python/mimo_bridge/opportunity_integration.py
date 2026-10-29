# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Completed learner progress, measured in a fixed acting encoder's coordinates."""
from collections import deque
import copy
import hashlib
import numpy as np
import torch
from lem.model.learning_opportunity import LearningOpportunity,CONTRACT
from .continuation_state import tree


ANCHOR_IDENTITY='per_body_prediction_contents_v2'


def anchor_identities(inputs,command,target):
    """Same prediction problem and outcome retain their earned best error.

    Elapsed observation time verifies causality separately. It cannot make an
    old lesson new. Other bodies cannot change this body's evidence identity.
    Context, achieved pose and memory bias are real predictor inputs and must
    distinguish otherwise identical sensory frames.
    """
    def update(digest,value,body):
        if torch.is_tensor(value):
            sample=value[body].detach().cpu().contiguous()
            digest.update(str((str(sample.dtype),tuple(sample.shape))).encode())
            digest.update(sample.view(torch.uint8).numpy().tobytes())
        elif isinstance(value,dict):
            for key,item in sorted(value.items()):
                digest.update(repr(key).encode());update(digest,item,body)
        else:digest.update(repr(value).encode())
    records=[]
    actual_inputs={key:inputs[key] for key in ('tokens','context','achieved','memory_bias','source_mask')}
    for body in range(len(command)):
        digest=hashlib.sha256(ANCHOR_IDENTITY.encode())
        update(digest,dict(inputs=actual_inputs,command=command,target=target),body)
        records.append(digest.hexdigest())
    return records


class OpportunityAudit:
    def __init__(self,model,configuration,criterion):
        from lem.model.forward_model import ForwardModel
        from lem.model.prediction_heads import PredictionHeads
        from .acting_snapshot import preserve_rng
        self.model=model;self.config=copy.deepcopy(configuration);self.criterion=criterion
        if configuration.get('contract')!=CONTRACT:raise ValueError('unknown learning-opportunity contract')
        self.every=int(configuration['audit_every']);self.capacity=int(configuration['anchor_capacity'])
        if self.every<1 or self.capacity<1:raise ValueError('invalid declared audit budget')
        self.meter=LearningOpportunity(configuration['window'],configuration['effect_floors'],
                                      configuration['calibration'],configuration.get('groups'),require_retention=True)
        with preserve_rng(model.device):
            self.reference=ForwardModel(copy.deepcopy(model.config)).to(model.device)
            self.reference_heads=PredictionHeads(copy.deepcopy(model.config)).to(model.device)
        self.reference.load_state_dict(model.forward_model.state_dict());self.reference_heads.load_state_dict(model.prediction_heads.state_dict())
        self.reference.eval();self.reference_heads.eval()
        for module in (self.reference,self.reference_heads):
            for parameter in module.parameters():parameter.requires_grad_(False)
        self.pending=None;self.anchors=deque(maxlen=self.capacity);self.retention_panel=[]
        self.completed=0;self.last={}

    def healthy(self):
        target=self.model.self_history.targets;valid=self.model.self_history.valid
        if target is None or valid is None:return None
        kind=self.model._self_percept['kind']
        minimum_capacity=1. if self.criterion.minimum_capacity is None else self.criterion.minimum_capacity
        threshold=torch.where(kind==1,minimum_capacity,self.criterion.minimum_resolution)
        return ((target[...,0]>=threshold)|~valid).all(-1)

    def health_record(self):
        target=self.model.self_history.targets;valid=self.model.self_history.valid
        if target is None or valid is None:return dict(available_bodies=0,reason='no self observation')
        kind=self.model._self_percept['kind'];q=target[...,0]
        cap=1. if self.criterion.minimum_capacity is None else self.criterion.minimum_capacity
        skin_bad=(q<self.criterion.minimum_resolution)&valid&(kind==0)
        motor_bad=(q<cap)&valid&(kind==1)
        rows=[]
        profile=self.model.self_intention.receipt['body_function_profile']
        for body in range(len(q)):
            reasons=[]
            if skin_bad[body].any():reasons.append('local sensing below declared distinction requirement')
            if motor_bad[body].any():reasons.append('motor capacity below measured operating requirement')
            skin=q[body][valid[body]&(kind[body]==0)]
            motor=q[body][valid[body]&(kind[body]==1)]
            bits=torch.round(profile['minimum_bits']+(profile['maximum_bits']-profile['minimum_bits'])*skin)
            quantum=profile['pressure_range_pa']/(2.**bits-1.)
            rows.append(dict(eligible=not reasons,denied_reasons=reasons,
                below_resolution_sites=int(skin_bad[body].sum()),below_capacity_groups=int(motor_bad[body].sum()),
                minimum_resolution_index=None if not len(skin) else float(skin.min()),
                minimum_motor_capacity=None if not len(motor) else float(motor.min()),
                worst_pressure_quantum_pa=None if not len(quantum) else float(quantum.max())))
        return dict(available_bodies=sum(r['eligible'] for r in rows),body_count=len(rows),bodies=rows,
                    meaning='Resolution index controls quantization bits; it is not a fraction of retained information')

    def inputs(self):
        model=self.model
        return dict(tokens=tree(model.last_tokens,model.device),context=tree(model.last_context,model.device),
            achieved=tree(model.last_achieved,model.device),memory_bias=tree(model._memory_bias_current,model.device),
            source_mask=copy.deepcopy(model._last_source_mask),time=tree(model.self_history.time,model.device))

    @torch.no_grad()
    def predict(self,forward,heads,inputs,commands):
        with torch.autocast(device_type=self.model.device.type,enabled=False):
            predictions,_=forward.predict_from_action(None,commands,self.model.encoder_registry,heads,
                achieved_state=inputs['achieved'],state_tokens=inputs['tokens'],
                context_buffer=inputs['context'],memory_bias=inputs['memory_bias'])
        return {k:v.detach().float() for k,v in predictions.items()}

    @torch.no_grad()
    def prepare(self,learner,executed_action,step):
        """Only past actor inputs and learner weights before the next observation.

        The world has completed the command, which includes its actual gaze
        overrides. Neither the learner nor this method has consumed the next
        observation. Its target is encoded on the next actor step.
        """
        from .lem_model import command_for_encoder
        if step%self.every or self.model.last_tokens is None:return
        if self.pending is not None:raise ValueError('a prospective audit target was skipped')
        inputs=self.inputs();command=command_for_encoder(torch.as_tensor(executed_action,device=self.model.device,dtype=torch.float32))
        if len(command)>1:wrong=command.roll(1,0)
        else:
            wrong=inputs['achieved'][...,:9].clone()
            # Achieved position is already normalized to the command encoder.
            wrong[...,7:9]=command[...,7:9]
        prediction=self.predict(learner.forward_model,learner.prediction_heads,inputs,command)
        shuffled=self.predict(learner.forward_model,learner.prediction_heads,inputs,wrong)
        reference=self.predict(self.reference,self.reference_heads,inputs,command)
        self.pending=dict(inputs=tree(inputs),command=tree(command),prediction=tree(prediction),
            shuffled=tree(shuffled),reference=tree(reference),healthy=tree(self.healthy()),step=int(step))
        # Re-test fixed anchors with current weights and unchanged encoder,
        # context, action and target. Historical best forbids relearning credit.
        measured={}
        for anchor in self.anchors:
            pred=self.predict(learner.forward_model,learner.prediction_heads,
                              tree(anchor['inputs'],self.model.device),tree(anchor['command'],self.model.device))
            measured[anchor['id']]={modality:(pred[modality].cpu()-target).square().mean(-1)
                                   for modality,target in anchor['target'].items()}
            for modality,target in anchor['target'].items():
                errors=measured[anchor['id']][modality]
                for body,error in enumerate(errors):
                    self.meter.observe_anchor(body,modality,anchor['body_ids'][body],reference=anchor['reference_error'][modality][body],
                                              prediction=float(error),healthy=bool(self.healthy()[body]))
        if len(self.retention_panel)==self.capacity:
            # These examples never rotate within one fixed actor/encoder. A
            # later reference refresh or new timestamp cannot erase mastery.
            panel_id=hashlib.sha256(''.join(a['id'] for a in self.retention_panel).encode()).hexdigest()
            grouped={}
            for anchor in self.retention_panel:
                if anchor['id'] not in measured:
                    pred=self.predict(learner.forward_model,learner.prediction_heads,
                        tree(anchor['inputs'],self.model.device),tree(anchor['command'],self.model.device))
                    measured[anchor['id']]={m:(pred[m].cpu()-target).square().mean(-1)
                                            for m,target in anchor['target'].items()}
                for modality,errors in measured[anchor['id']].items():
                    grouped.setdefault(modality,[]).append((errors,anchor['reference_error'][modality]))
            for modality,pairs in grouped.items():
                prediction=torch.stack([p[0] for p in pairs]).mean(0)
                reference=torch.tensor([p[1] for p in pairs]).mean(0)
                for body in range(len(prediction)):
                    self.meter.observe_retention(body,modality,panel_id,prediction=float(prediction[body]),
                        reference=float(reference[body]))
        records=list(self.meter.retention.values())
        if records and all(r['preserved'] for r in records) and any(r['improved'] for r in records):
            # A rolling clock cannot replace a competent reference with a
            # regressed predictor and then pay for catching up with it.
            self.reference.load_state_dict(learner.forward_model.state_dict())
            self.reference.invalidate_embedding_cache()
            self.reference_heads.load_state_dict(learner.prediction_heads.state_dict())

    @torch.no_grad()
    def complete(self):
        pending=self.pending
        if pending is None:return
        now=self.model.self_history.time.cpu();old=pending['inputs']['time'].cpu()
        if now.shape!=old.shape or not (now>old).all():raise ValueError('audit outcome does not follow its inputs')
        healthy=self.healthy().cpu() & pending['healthy'].cpu()
        target={};reference_error={};comparisons={}
        for modality,prediction in pending['prediction'].items():
            if (modality not in self.meter.floors or pending['inputs']['source_mask'].get(modality)!='real'
                    or self.model._last_source_mask.get(modality)!='real'):continue
            actual=self.model.last_tokens[modality].float().cpu()
            error=(prediction-actual).square().mean(-1)
            reference=(pending['reference'][modality]-actual).square().mean(-1)
            shuffled=(pending['shuffled'][modality]-actual).square().mean(-1)
            persistence=(pending['inputs']['tokens'][modality]-actual).square().mean(-1)
            target[modality]=actual.clone();reference_error[modality]=reference.tolist()
            comparisons[modality]=dict(prediction_mse=error.tolist(),action_control_mse=shuffled.tolist(),
                                      persistence_mse=persistence.tolist(),eligible=healthy.tolist())
            for body in range(len(actual)):
                self.meter.observe_fresh(body,modality,prediction=float(error[body]),shuffled=float(shuffled[body]),
                    persistence=float(persistence[body]),time=float(now[body]),healthy=bool(healthy[body]))
        body_ids=anchor_identities(pending['inputs'],pending['command'],target)
        identity=hashlib.sha256(''.join(body_ids).encode()).hexdigest()
        anchor=dict(id=identity,body_ids=body_ids,inputs=pending['inputs'],command=pending['command'],target=target,
                    reference_error=reference_error)
        # Exact duplicate batches do not crowd out distinct retained examples.
        if not any(row['id']==identity for row in self.anchors):self.anchors.append(anchor)
        if len(self.retention_panel)<self.capacity and not any(row['id']==identity for row in self.retention_panel):
            self.retention_panel.append(anchor)
        self.pending=None;self.completed+=1
        self.last=dict(contract=CONTRACT,completed_comparisons=self.completed,retained_anchors=len(self.anchors),
                       fixed_retention_examples=len(self.retention_panel),
                       audit_health=self.health_record(),latest_comparisons=comparisons,
                       eligibility=[{m:self.meter.explain(b,m) for m in self.meter.floors} for b in range(len(now))],
                       distinction_progress=[{m:self.meter.available(b,m)[0] for m in self.meter.floors}
                                             for b in range(len(now))])

    @torch.no_grad()
    def score(self,commands,eligible):
        # Only first-step sensory consequences are used: pretending a predicted
        # token was a known future physical pose would overstate this forecast.
        b,k,_,a=commands.shape
        if not any(self.meter.available(body,modality)[0]>0 for body in range(b) for modality in self.meter.floors):
            return torch.zeros(b,k,device=commands.device,dtype=torch.float32)
        inputs=self.inputs()
        flat={key:({m:v[:,None].expand(b,k,*v.shape[1:]).reshape(b*k,*v.shape[1:]) for m,v in value.items()}
                   if key=='tokens' else None if value is None else
                   value[:,None].expand(b,k,*value.shape[1:]).reshape(b*k,*value.shape[1:]))
              for key,value in inputs.items() if key in ('tokens','context','achieved','memory_bias')}
        prediction=self.predict(self.model.forward_model,self.model.prediction_heads,flat,
                                commands[:,:,0].reshape(b*k,-1,9))
        prediction={m:v.reshape(b,k,-1).cpu().numpy() for m,v in prediction.items()}
        result=[];healthy=self.healthy().cpu().tolist()
        for body in range(b):
            p={m:v[body] for m,v in prediction.items()}
            result.append(self.meter.score(body,p,{m:v[1] for m,v in p.items()},
                                           eligible=eligible[body].cpu().numpy(),healthy=healthy[body]))
        return torch.as_tensor(np.stack(result),device=commands.device,dtype=torch.float32)

    def reset(self):
        self.pending=None;self.anchors.clear();self.meter.reset_episode();self.last={}

    def state_dict(self):
        return tree(dict(config=self.config,anchor_identity=ANCHOR_IDENTITY,meter=self.meter.state_dict(),pending=self.pending,anchors=self.anchors,
            retention_panel=self.retention_panel,
            reference=self.reference.state_dict(),reference_heads=self.reference_heads.state_dict(),
            completed=self.completed,last=self.last))

    def load_state_dict(self,state):
        if state['config']!=self.config:raise ValueError('opportunity audit calibration changed')
        if state.get('anchor_identity')!=ANCHOR_IDENTITY:
            raise ValueError('old timestamp-based opportunity evidence requires explicit migration; it cannot certify new learning')
        self.meter.load_state_dict(state['meter']);self.pending=tree(state['pending']);self.anchors=tree(state['anchors'])
        self.retention_panel=tree(state['retention_panel'])
        self.reference.load_state_dict(state['reference']);self.reference_heads.load_state_dict(state['reference_heads'])
        self.reference.invalidate_embedding_cache()
        self.completed=state['completed'];self.last=copy.deepcopy(state['last'])
