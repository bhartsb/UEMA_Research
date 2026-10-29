# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""An independently qualified acting brain beside a continuing learner.

The actor observes every actual frame and executed command with its own encoders,
recurrent state, intention and random stream. Learner updates cannot change it.
The physical episode checkpoint contains both histories. Replacing an actor is
an explicit qualification operation, never a silent copy of learner weights.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import random

import numpy as np
import torch

from . import continuation_state as continuation
from .self_intention import SelfIntention, fingerprint, perception_fingerprint, perception_modules

CONTRACT = 'frozen_acting_snapshot_v1'


def configuration_fingerprint(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def model_execution_description(model):
    from .execution_contract import loaded_execution
    names=('active','handle_names','body_layout','contact_profile','self_state_mode','self_prediction_mode',
           'top_down_gate','intention_enabled','intention_every','intention_execute','think_harder',
           'smooth_actions','goal_gaze_hold','goal_revalue','goal_locality','curiosity_goal',
           'agency_drive','agency_weight','movement_sensor_mod','movement_param_mod','vision_gate_mode',
           'plan_horizon_mode','body_schema_updates','prospective_wellness','prospective_wellness_w')
    values={k:getattr(model,k,None) for k in names}
    values.update(config=model.config,wellness_mode=str(model.wellness_mode),dtype=str(model.dtype),
                  loaded_execution=loaded_execution(model))
    return values


def model_execution_fingerprint(model):
    return configuration_fingerprint(model_execution_description(model))


def rng_state(device):
    return dict(torch=torch.get_rng_state().clone(),numpy=copy.deepcopy(np.random.get_state()),
                python=random.getstate(),cuda=(torch.cuda.get_rng_state_all() if device.type=='cuda' else None))


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


@contextmanager
def preserve_rng(device):
    saved=rng_state(device)
    try:yield
    finally:restore_rng(saved)


def runtime(model):
    reg=model.latent_regularizer
    return dict(schemas=[continuation.tree(s.state_dict()) for s in model._schemas],
                hself=None if model._hself is None else continuation.tree(model._hself.state_dict()),
                latent=None if reg is None else dict(mean=continuation.tree(reg.running_mean),
                    variance=continuation.tree(reg.running_var),initialized=reg._initialized))


def restore_runtime(model,state):
    from lem.model.body_schema import DifferentiableBodySchema
    from mimo_bridge.lem_model import SELF_OBJECTS
    # THE MAP IS BUILT THE WAY THE MODEL BUILDS IT, outside things included (UEMA 11.2).
    # Built without them it carries none of their buffers, and a strict load of today's
    # state raised on the extra keys while a snapshot taken before they existed raised on
    # the missing ones - a frozen acting snapshot could not be restored in either direction.
    # strict=False is what lets the two ages meet: an older snapshot leaves the outside
    # things at their starting zero, which is "nothing has been taken into the self yet".
    while len(model._schemas)<len(state['schemas']):
        model._schemas.append(DifferentiableBodySchema(
            n_handles=len(model.handle_names),object_names=SELF_OBJECTS).to(model.device))
    for module,data in zip(model._schemas,state['schemas']):
        missing,unexpected=module.load_state_dict(data,strict=False)
        if unexpected:
            raise ValueError('acting snapshot body map carries unknown entries: %s'%sorted(unexpected))
    if state['hself'] is not None:
        if model._hself is None:raise ValueError('acting snapshot requires retained self evidence')
        model._hself.load_state_dict(continuation.tree(state['hself'],model.device))
    if state['latent'] is not None:
        reg=model.latent_regularizer
        if reg is None:raise ValueError('acting snapshot requires retained latent reference')
        reg.running_mean=state['latent']['mean'].to(model.device).clone()
        reg.running_var=state['latent']['variance'].to(model.device).clone()
        reg._initialized=state['latent']['initialized']


class FrozenActingSnapshot:
    def __init__(self,model,receipt,configuration,*,qualification=False):
        self._require_supported_hand_stage(model)
        self.model=model
        self.configuration=copy.deepcopy(configuration)
        self.qualification=bool(qualification)
        self.receipt=copy.deepcopy(receipt)
        if not qualification and (receipt.get('actor_contract')!=CONTRACT or
                receipt.get('acting_configuration_sha256')!=configuration_fingerprint(configuration)):
            raise ValueError('acting snapshot requires qualification of the exact execution configuration')
        model.self_intention=SelfIntention('qualification' if qualification else 'on',self.receipt)
        # Evaluation mode belongs to this versioned actor contract and is used in
        # qualification too. Gradients through candidate actions remain available.
        for module in perception_modules(model).values():
            module.eval()
            for parameter in module.parameters():parameter.requires_grad_(False)
        self.random=rng_state(model.device)
        self.born_step=int(model.step_count)
        self.steps=0
        self.last={}
        self.pending=None
        self.opportunity=None
        if self.receipt.get('learning_opportunity') is not None:
            from .opportunity_integration import OpportunityAudit
            self.opportunity=OpportunityAudit(model,self.receipt['learning_opportunity'],model.self_intention.criterion)
        model.learning_opportunity_audit=self.opportunity
        self.verify_weights()

    @staticmethod
    def _require_supported_hand_stage(model):
        if getattr(model,'hand_goals','off')!='off':
            raise ValueError('Frozen acting snapshots do not yet bind the calibrated hand-goal stage; '
                             'keep self-choice off until the actor contract is extended and qualified')

    def verify_weights(self):
        self._require_supported_hand_stage(self.model)
        if (fingerprint(self.model.self_state)!=self.receipt.get('self_weights_sha256') or
                perception_fingerprint(self.model)!=self.receipt.get('perception_weights_sha256')):
            raise ValueError('acting snapshot does not match qualified weights')
        actual=model_execution_description(self.model)
        if configuration_fingerprint(actual)!=self.receipt.get('model_execution_sha256'):
            expected=self.receipt.get('model_execution_description')
            changed=[] if expected is None else [name for name in sorted(set(actual)|set(expected))
                if actual.get(name)!=expected.get(name)]
            raise ValueError('acting model execution differs from qualified configuration; '
                             +'changed fields: '+(', '.join(changed) if changed else 'legacy receipt has no descriptor'))

    @classmethod
    def from_learner(cls,learner,factory,receipt,configuration,*,qualification=False):
        """Initially branch matching weights AND their matching causal history."""
        cls._require_supported_hand_stage(learner)
        with preserve_rng(learner.device):
            state=continuation.capture_learner(learner,include_replay=False)
            model=factory()
            model.config.clear();model.config.update(copy.deepcopy(learner.config))
            for name,source in perception_modules(learner).items():
                target=perception_modules(model).get(name)
                if target is None:raise ValueError('acting snapshot layout differs: '+name)
                target.load_state_dict(source.state_dict(),strict=True)
            continuation.restore_learner(model,state)
            restore_runtime(model,runtime(learner))
            model.step_count=learner.step_count
            return cls(model,receipt,configuration,qualification=qualification)

    def step(self,frames,dev,*,step_index,executed_action,executed_order_id=None):
        # This is checked before inference as well as by the chooser. An encoder
        # change cannot even produce an unqualified fallback command unnoticed.
        self._require_supported_hand_stage(self.model)
        if not self.model.self_intention.validation_matches(self.model,frames):
            raise ValueError('acting snapshot weight or physical contract changed')
        if model_execution_fingerprint(self.model)!=self.receipt.get('model_execution_sha256'):
            raise ValueError('acting model execution configuration changed')
        with preserve_rng(self.model.device):
            restore_rng(self.random)
            # The trainer may hold, constrain or externally override a proposal.
            # Smooth the next order from what reached the body, not from the
            # actor's discarded proposal on the intervening sensing steps.
            if executed_action is not None:
                self.model._prev_mfa=torch.as_tensor(executed_action,device=self.model.device,dtype=torch.float32).clone()
            result=self.model.step(frames,dev,train=False,step_index=step_index,
                executed_action=executed_action,executed_order_id=executed_order_id)
            self.random=rng_state(self.model.device)
        if not self.model.self_intention.validation_matches(self.model,frames):
            raise ValueError('acting inference changed a qualified dependency')
        self.steps+=1
        if self.opportunity is not None:self.opportunity.complete()
        decision=copy.deepcopy(self.model.self_intention.last)
        self.last=dict(contract=CONTRACT,qualified=not self.qualification,
            source='copied qualification trial' if self.qualification else 'validated frozen brain',
            snapshot_step=self.born_step,observations=self.steps,learner_updates_independent=True,
            new_self_choice=bool(decision.get('executed') and decision.get('decision_step')==step_index),
            decision=decision)
        if self.opportunity is not None:self.last['learning_opportunity']=copy.deepcopy(self.opportunity.last)
        if self.pending is not None:
            self.pending.step(frames,dev,step_index=step_index,executed_action=executed_action,
                              executed_order_id=executed_order_id)
            self.last['candidate']=dict(snapshot_step=self.pending.born_step,observations=self.pending.steps,
                                       applied_to_body=False)
        return result

    def reset_transition_state(self):
        self.model.reset_transition_state()
        if self.model.movement is not None:self.model.movement.reset()
        self.model.self_intention.last={}
        self.last={}
        if self.pending is not None:self.pending.reset_transition_state()
        if self.opportunity is not None:self.opportunity.reset()

    def prepare_learning_audit(self,learner,executed_action,step):
        if self.opportunity is not None:
            with preserve_rng(self.model.device):
                self.opportunity.prepare(learner,executed_action,step)
        if self.pending is not None:self.pending.prepare_learning_audit(learner,executed_action,step)

    def stage(self,learner,factory,calibration):
        """Keep acting while a new frozen candidate follows the real experience."""
        if self.pending is not None:raise ValueError('an observing replacement already awaits its result')
        if self.qualification:raise ValueError('the initial candidate already awaits qualification')
        receipt=calibration_for_candidate(learner,calibration,self.configuration)
        self.pending=self.from_learner(learner,factory,receipt,self.configuration,qualification=True)
        if self.opportunity is not None and self.pending.opportunity is not None:
            self.pending.opportunity.meter.archive_previous(self.opportunity.meter.state_dict(),
                                                           fingerprint(self.model.encoder_registry))

    def promote(self,receipt):
        """Adopt measured evidence without changing this observer's live history.

        Qualification can run on another copy of these exact weights. This
        candidate keeps observing the real episode throughout that comparison.
        A refused replacement leaves the existing actor and its receipt intact.
        """
        if self.pending is not None:
            return self.pending.promote(receipt)
        self.verify_weights()
        chooser=SelfIntention('on',receipt)
        calibrated=('minimum_resolution','minimum_capacity','maximum_uncertainty','candidate_position_fraction',
                    'candidate_rotation_fraction','order_steps','orders','step_seconds','body_function',
                    'body_function_profile','motor_contract')
        calibrated=calibrated+('learning_opportunity',)
        if any(receipt.get(k)!=self.receipt.get(k) for k in calibrated):
            raise ValueError('promotion changed the candidate calibrated decision rule')
        if (receipt.get('actor_contract')!=CONTRACT or
                receipt.get('acting_configuration_sha256')!=configuration_fingerprint(self.configuration) or
                receipt.get('model_execution_sha256')!=model_execution_fingerprint(self.model) or
                receipt.get('self_weights_sha256')!=fingerprint(self.model.self_state) or
                receipt.get('perception_weights_sha256')!=perception_fingerprint(self.model)):
            raise ValueError('promotion evidence describes a different acting snapshot')
        self.receipt=copy.deepcopy(receipt)
        self.model.self_intention=chooser
        self.qualification=False
        self.last=dict(contract=CONTRACT,qualified=True,source='validated frozen brain',
                       snapshot_step=self.born_step,observations=self.steps,
                       new_self_choice=False,decision=dict(status='qualified; next decision pending',executed=False))
        return self

    def state_dict(self):
        self.verify_weights()
        with preserve_rng(self.model.device):
            restore_rng(self.random)
            causal=continuation.capture_learner(self.model,include_replay=False)
        return dict(contract=CONTRACT,receipt=copy.deepcopy(self.receipt),
            configuration=copy.deepcopy(self.configuration),qualification=self.qualification,
            modules={k:continuation.tree(m.state_dict()) for k,m in perception_modules(self.model).items()},
            causal=causal,runtime=runtime(self.model),random=continuation.tree(self.random),
            model_config=copy.deepcopy(self.model.config),
            pending=None if self.pending is None else self.pending.state_dict(),
            opportunity=None if self.opportunity is None else self.opportunity.state_dict(),
            born_step=self.born_step,step_count=self.model.step_count,steps=self.steps,last=copy.deepcopy(self.last),
            chooser_last=copy.deepcopy(self.model.self_intention.last))

    @classmethod
    def restore(cls,state,factory,device,*,allow_qualification=False):
        if state.get('contract')!=CONTRACT:raise ValueError('incompatible acting snapshot')
        if state['qualification'] and not allow_qualification:
            raise ValueError('an unqualified trial cannot resume as a production actor')
        with preserve_rng(torch.device(device)):
            model=factory()
            model.config.clear();model.config.update(copy.deepcopy(state['model_config']))
            if set(perception_modules(model))!=set(state['modules']):
                raise ValueError('acting snapshot module layout differs')
            for name,module in perception_modules(model).items():module.load_state_dict(state['modules'][name],strict=True)
            continuation.restore_learner(model,state['causal'])
            restore_runtime(model,state['runtime'])
            model.step_count=state['step_count']
            actor=cls(model,state['receipt'],state['configuration'],qualification=state['qualification'])
            actor.born_step=state['born_step'];actor.steps=state['steps'];actor.last=copy.deepcopy(state['last'])
            actor.random=continuation.tree(state['random'])
            actor.model.self_intention.last=copy.deepcopy(state['chooser_last'])
            if state.get('opportunity') is not None:
                if actor.opportunity is None:raise ValueError('saved learning-opportunity perception cannot be discarded')
                actor.opportunity.load_state_dict(state['opportunity'])
            if state.get('pending') is not None:
                actor.pending=cls.restore(state['pending'],factory,device,allow_qualification=True)
            return actor


def calibration_for_candidate(model,calibration,configuration):
    """Describe a copied candidate; this function supplies NO behavioral pass."""
    receipt=copy.deepcopy(calibration)
    description=model_execution_description(model)
    receipt.update(actor_contract=CONTRACT,acting_configuration_sha256=configuration_fingerprint(configuration),
        self_weights_sha256=fingerprint(model.self_state),perception_weights_sha256=perception_fingerprint(model),
        model_execution_sha256=configuration_fingerprint(description),model_execution_description=description)
    for key in ('choice_gain_lower','action_effect_gain_lower','retention_pass','replay_verified','held_out_worlds',
                'acquisition_gain_lower','viability_gain_lower','qualification_route','acquisition_metric',
                'acquisition_comparator','prediction_error_upper','decision_effect_separation_lower','action_prediction_scale'):
        receipt.pop(key,None)
    receipt['schema']='unqualified_self_choice_candidate_v1'
    return receipt
