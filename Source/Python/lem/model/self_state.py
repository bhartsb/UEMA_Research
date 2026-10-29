# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Learned local self perception and action-conditioned future functioning.

The existing brain supplies sensory representations. This extension retains local
sites, temporal state and a learned relation memory; it never names objects or actions.
Numerical function decoders are grounding tests, not the definition of wellness.
"""
from collections import deque
import copy
import logging
import torch
from torch import nn
import torch.nn.functional as F

log=logging.getLogger(__name__)

STATE_FIELDS=3
SITE_INPUT=3+8+STATE_FIELDS  # location, experienced touch, observed local function
# Actuator groups whose measured availability joins the skin rows. The producer is
# env.py's len(self.proprio.fatigue.cap), a property of the body's actuator set. A
# layout reporting a different count raises 'invalid actual motor capacity' below,
# and a checkpoint trained at this count cannot be migrated as things stand, because
# capacity_identity's table is a saved tensor. Naming it gives that migration, and a
# receipt entry beside body_layout, one place to hang.
MOTOR_GROUPS=10

# The self-percept is STORED and RECALLED: perceive() writes `belief` into the relation
# memory and the next perceive() attends over those entries to build the next belief. A
# gain above 1 anywhere in that path compounds every step, with nothing in the arithmetic
# to stop it. Measured on H84 (2026-09-20): healthy stored percepts ran |x| 1.29 to 1.49
# over 128 entries; on the diverged run the same 128 entries ran 2.1e+34 to 3.8e+35,
# growing about 1.023x per step, while the inputs stayed ordinary (context 4.13 vs 4.27,
# sites 81 vs 144). At that size one body's forecast overflowed to non-finite, the guard
# stopped the run, and 18 automatic resumes re-entered the same weights: 6.4 hours for
# 418 steps. The belief is rescaled - direction kept, gradient kept - when it passes this
# limit, which is 5x the largest healthy value ever measured on this lineage, so healthy
# training never reaches it.
BELIEF_LIMIT=8.0


class SelfState(nn.Module):
    def __init__(self,d_model,n_modalities,action_size,prediction_mode='legacy'):
        super().__init__()
        if prediction_mode not in ('legacy','grounded_residual'):
            raise ValueError('unknown self prediction mode')
        self.prediction_mode=prediction_mode
        self.d_model=d_model;self.n_modalities=n_modalities;self.action_size=action_size
        # a plain counter, deliberately NOT a buffer: a new checkpoint key would refuse
        # every resume from a snapshot written before today
        self.belief_bounded_count=0
        # None (default, every run before 2026-09-26): the recall's attention is used as the
        # softmax produced it. A number (LemLite sets ATTENTION_PEAK_LIMIT when
        # --self-percept-release on): no stored percept may carry more than that share of a
        # body's recall. A plain attribute, not a buffer, for the same resume reason as the
        # counter above. ATTENTION_PEAK_LIMIT says why.
        self.attention_peak_limit=None
        self.last_attention_capped=0;self.last_attention_peak=None
        self.site_encoder=nn.Sequential(nn.Linear(SITE_INPUT,d_model),nn.LayerNorm(d_model),nn.GELU())
        self.capacity_encoder=nn.Linear(1,d_model)
        self.capacity_identity=nn.Embedding(MOTOR_GROUPS,d_model)
        self.regime_encoder=nn.Embedding(2,d_model)
        self.context_encoder=nn.Linear(n_modalities*d_model,d_model)
        self.internal_encoder=nn.Linear(d_model,d_model)
        self.recurrent=nn.GRUCell(3*d_model+1,d_model)
        self.relation_query=nn.Linear(d_model,d_model)
        self.relation_key=nn.Linear(d_model,d_model)
        self.relation_value=nn.Linear(d_model,d_model)
        self.body_context=nn.Linear(2*d_model,d_model)
        self.present=nn.Linear(d_model,STATE_FIELDS)
        self.action_encoder=nn.Linear(action_size+1,d_model)
        self.transition=nn.GRUCell(d_model,d_model)
        self.future=nn.Sequential(nn.Linear(2*d_model,d_model),nn.GELU(),nn.Linear(d_model,2*STATE_FIELDS))
        self.residual=nn.Linear(d_model,n_modalities*d_model)
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        # Appended parameters preserve every earlier tensor name and order. The
        # old belief is retained; observed function is a grounding channel, not
        # a replacement for that learned perception or a supplied action target.
        if prediction_mode == 'grounded_residual':
            self.grounded=GroundedDynamics(d_model,action_size)

    def perceive(self,sites,valid,context,previous=None,internal=None,dt=None,relations=None,
                 capacity=None,capacity_valid=None,regime=None,spatial=None,spatial_valid=None,function_extent=None,surface_moments=None):
        b,n,_=sites.shape
        if valid.shape!=(b,n) or context.shape!=(b,self.n_modalities*self.d_model):
            raise ValueError('self perception must preserve body/site layout')
        if not torch.isfinite(sites).all() or not torch.isfinite(context).all():
            raise ValueError('invalid self perception')
        if not valid.any(-1).all():
            raise ValueError('each body needs at least one observed site')
        x=self.site_encoder(sites.float())
        if function_extent is not None:
            if self.prediction_mode!='grounded_residual' or function_extent.shape!=(b,n,2) or not torch.isfinite(function_extent).all() or ((function_extent<0)|(function_extent>1)).any():
                raise ValueError('invalid local function extent')
            x=x+self.grounded.scope_encoder(function_extent.float())
        if surface_moments is not None:
            if self.prediction_mode!='grounded_residual' or surface_moments.shape!=(b,n,20) or not torch.isfinite(surface_moments).all():
                raise ValueError('invalid local surface moments')
            # Centres in metres, covariances in square metres, validity flags.
            scale=x.new_tensor(([.1]*3+[.01]*6)*2+[1.,1.])
            x=x+self.grounded.surface_encoder(surface_moments.float()/scale)
        kind=torch.zeros_like(valid,dtype=torch.long)
        if capacity is not None:
            if (capacity.shape!=(b,MOTOR_GROUPS) or capacity_valid is None or capacity_valid.shape!=(b,MOTOR_GROUPS)
                    or not torch.isfinite(capacity).all() or ((capacity<0)|(capacity>1)).any()):
                raise ValueError('invalid actual motor capacity')
            cx=self.capacity_encoder(capacity.float()[...,None])+self.capacity_identity(torch.arange(MOTOR_GROUPS,device=x.device))[None]
            x=torch.cat([x,cx],1);valid=torch.cat([valid,capacity_valid],1)
            kind=torch.cat([kind,torch.ones_like(capacity_valid,dtype=torch.long)],1)
        weights=valid.float().unsqueeze(-1)
        pooled=(x*weights).sum(1)/weights.sum(1).clamp_min(1)
        if regime is not None:
            if regime.shape!=(b,) or ((regime<0)|(regime>1)).any():raise ValueError('invalid experienced body regime')
            pooled=pooled+self.regime_encoder(regime.long())
        old=torch.zeros_like(pooled) if previous is None else previous
        inter=torch.zeros_like(pooled) if internal is None else internal
        delta=torch.zeros(b,1,device=x.device) if dt is None else dt.reshape(b,1)
        if not torch.isfinite(delta).all() or (delta < 0).any():
            raise ValueError('invalid elapsed simulation time')
        h=self.recurrent(torch.cat([pooled,self.context_encoder(context.float()),
                                   self.internal_encoder(inter.float()),torch.log1p(delta)],-1),old.float())
        recall=torch.zeros_like(h)
        self.last_attention_capped=0;self.last_attention_peak=None
        if relations is not None and relations.shape[1]:
            # Entries are earlier learned percepts. Attention encodes relations to
            # current context; it receives no object role or desired behavior label.
            scores=torch.einsum('bd,btd->bt',self.relation_query(h),self.relation_key(relations))/self.d_model**.5
            attention=scores.softmax(-1)
            if self.attention_peak_limit is not None:
                # --self-percept-release on (ATTENTION_PEAK_LIMIT says why): the peak before
                # the cap is kept for the stats; a body whose peak is inside the limit keeps
                # its attention value for value.
                peak=attention.detach().amax(-1)
                self.last_attention_peak=peak
                over=peak>self.attention_peak_limit
                self.last_attention_capped=int(over.sum())
                if self.last_attention_capped:
                    attention=torch.where(over[:,None],cap_attention(scores,self.attention_peak_limit),attention)
            recall=torch.einsum('bt,btd->bd',attention,self.relation_value(relations))
        belief=self.body_context(torch.cat([h,recall],-1))
        size=belief.detach().abs().amax(-1,keepdim=True)
        # Each body's size BEFORE any rescale, a plain attribute like the counter below.
        # self_state_integration.observe() reads it right after the live call (the only call
        # whose percept is stored) to report belief_size and to decide the release.
        self.last_belief_size=size[:,0]
        if bool((size>BELIEF_LIMIT).any()):
            self.belief_bounded_count+=1
            log.warning('self perception: the stored self-percept reached %.4g, above the '
                        'limit of %.4g, and was rescaled; healthy on H84 was 1.4 to 2.0. '
                        'This is the runaway that stopped the run of 2026-09-20 - read '
                        'self_state.present_error and the self-state term of the loss.',
                        float(size.max()),BELIEF_LIMIT)
            belief=belief*torch.where(size>BELIEF_LIMIT,BELIEF_LIMIT/size.clamp_min(1e-6),
                                      torch.ones_like(size))
        local=x+belief[:,None]
        result=dict(state=h,belief=belief,local=local,present=self.present(local).sigmoid(),valid=valid,kind=kind,
                    bias=self.residual(belief).reshape(b,self.n_modalities,self.d_model))
        if self.prediction_mode == 'grounded_residual':
            observed=sites[...,-STATE_FIELDS:]
            observed_sites=sites
            if ((observed<0)|(observed>1)).any():raise ValueError('observed local function outside its physical range')
            if capacity is not None:
                observed=torch.cat([observed,torch.stack([capacity,torch.zeros_like(capacity),torch.zeros_like(capacity)],-1)],1)
                tail=sites.new_zeros(b,MOTOR_GROUPS,SITE_INPUT);tail[...,-STATE_FIELDS:]=observed[:,-MOTOR_GROUPS:]
                observed_sites=torch.cat([observed_sites,tail],1)
            result.update(learned_present=result['present'],observed=observed,present=observed,observed_sites=observed_sites)
            if spatial is not None:
                if spatial_valid is None:raise ValueError('spatial self observations require a validity mask')
                result.update(spatial=spatial,spatial_valid=spatial_valid)
        return result

    def forecast(self,percept,actions,dt):
        """Actions [body,horizon,flattened executed-command units], durations seconds."""
        b,t,a=actions.shape
        if (not t or a!=self.action_size or dt.shape!=(b,t) or (dt<=0).any()
                or not torch.isfinite(actions).all() or not torch.isfinite(dt).all()):
            raise ValueError('future self requires action sequence and positive simulation durations')
        if self.prediction_mode == 'grounded_residual':
            return self.grounded(percept,actions,dt)
        h=percept['belief'];local=percept['local'];means=[];variances=[]
        for j in range(t):
            h=self.transition(self.action_encoder(torch.cat([actions[:,j],torch.log1p(dt[:,j:j+1])],-1)),h)
            features=self.future[:2](torch.cat([local,h[:,None].expand(-1,local.shape[1],-1)],-1))
            head=self.future[2]
            mean=F.linear(features,head.weight[:STATE_FIELDS],head.bias[:STATE_FIELDS])
            # Fit residual uncertainty without letting trivially predictable
            # channels shrink variance by overwhelming the dynamics representation.
            log_variance=F.linear(features.detach(),head.weight[STATE_FIELDS:],head.bias[STATE_FIELDS:])
            means.append(mean.sigmoid());variances.append(log_variance.clamp(-9,3).exp())
        return torch.stack(means,1),torch.stack(variances,1)

    def present_loss(self,percept,target):
        """Retain a learned grounding audit even when current function is observed."""
        prediction=percept.get('learned_present',percept['present'])
        valid=percept['valid'][...,None].expand_as(target)
        error=(prediction-target).square()
        if self.prediction_mode == 'legacy':return error[valid].mean()
        # Group by physical function channel, then by unavailable/available
        # observation. The boundary is measured numeric equality, not a pain
        # cutoff. Healthy sites retain equal weight so local repair cannot hide
        # a whole-body downward bias. Different fields are never pooled first.
        return balanced_function_mean(error,target,valid)

    def learning_loss(self,mean,variance,target,valid,initial=None):
        if self.prediction_mode == 'legacy':return self.consequence_loss(mean,variance,target,valid)
        # Validate with the original contract before applying the local measure.
        self._validate_consequence(mean,variance,target,valid)
        mask=valid[...,None].expand_as(mean)
        error=(mean-target).square()
        nll=.5*(error.detach()/variance+variance.log())
        return balanced_function_mean(error+nll,target,mask,initial=initial)

    def paired_consequence_loss(self,mean,variance,target,valid):
        """Actual physical branches from one identical initial state only.

        The caller must have restored the same full world before every command.
        Ordinary bodies in an online minibatch are not counterfactual branches.
        Measured action variation sets the scale; zero-variation fields cannot
        manufacture an action lesson. Absolute and action-difference errors both
        contribute, so a correct difference plus a wrong common bias cannot win.
        """
        self._validate_consequence(mean,variance,target,valid)
        if mean.shape[0]<2:raise ValueError('paired lesson needs actual alternative actions')
        mask=valid[...,None].expand_as(mean)
        if not torch.equal(valid,valid[:1].expand_as(valid)):
            raise ValueError('physical branches must retain identical site validity')
        delta=target-target.mean(0,keepdim=True)
        prediction=mean-mean.mean(0,keepdim=True)
        losses=[]
        for field in range(STATE_FIELDS):
            selected=mask[...,field]
            signal=delta[...,field].square()[selected].mean()
            if signal<=64*torch.finfo(mean.dtype).eps**2:continue
            # Both terms share the measured action variance. Global offsets cost
            # prediction error even if differences happen to fit.
            field_mask=torch.zeros_like(mask);field_mask[...,field]=selected
            absolute=balanced_function_mean((mean-target).square()/signal,target,field_mask)
            contrast=(prediction[...,field]-delta[...,field]).square()[selected].mean()/signal
            losses.append(absolute+contrast)
        if not losses:raise ValueError('no measured action differences in paired lesson')
        uncertainty=.5*((mean-target).square().detach()/variance+variance.log())
        return torch.stack(losses).mean()+balanced_function_mean(uncertainty,target,mask)

    @staticmethod
    def _validate_consequence(mean,variance,target,valid):
        """The shape/finiteness/positivity contract, without the loss arithmetic.

        learning_loss and paired_consequence_loss want only these checks. Calling the
        whole loss to get them built and threw away a measured 0.328 ms per call on the
        run's own (8,20,63,3) shapes - 24.0% of learning_loss's 2.542 ms - three or four
        times per model step plus the rehearsal copy.
        """
        problems=[]
        if mean.shape!=variance.shape or mean.shape!=target.shape:
            problems.append('shapes disagree: mean %s variance %s target %s'
                            %(tuple(mean.shape),tuple(variance.shape),tuple(target.shape)))
        for name,tensor in (('mean',mean),('target',target),('variance',variance)):
            finite=torch.isfinite(tensor)
            if not finite.all():
                problems.append('%s carries %d non-finite of %d values'
                                %(name,int((~finite).sum()),tensor.numel()))
        if torch.isfinite(variance).all() and (variance<=0).any():
            problems.append('smallest variance %.3g is not above zero'%float(variance.min()))
        if not valid.any():
            problems.append('no future evidence is marked valid (0 of %d)'%valid.numel())
        if problems:
            # WHICH check failed, with its numbers. The bare sentence cost 18 crash-resume
            # cycles on 2026-09-20 (H84 segment 13, 6.4 hours) without saying whether the
            # evidence was missing, mis-shaped, non-finite or zero-variance.
            raise ValueError('invalid or absent future self-state evidence: '+'; '.join(problems))

    @staticmethod
    def consequence_loss(mean,variance,target,valid):
        SelfState._validate_consequence(mean,variance,target,valid)
        mask=valid.float()
        while mask.ndim<mean.ndim:mask=mask.unsqueeze(-1)
        mask=mask.expand_as(mean)
        # Proper variance-sensitive likelihood, with physical bounded means.
        error=(mean-target).square()
        loss=error+.5*(error.detach()/variance+variance.log())
        return (loss*mask).sum()/mask.sum().clamp_min(1)


# --self-percept-release on: the largest share of a body's recall one stored percept may carry.
# Why (docs/gain_of_function_v4/agents/self_percept_runaway_2026-09-26.md, second round): the recall
# path (body_context's recall half times relation_value) is a strong NEGATIVE feedback - its
# largest eigenvalues sit at -4.0 to -4.5 on the checkpoints from 344,400 to 407,032 and -2.1 at
# 415,075, with no real part above 0.02. Spread over the 128 stored percepts (healthy peak weight
# 0.009 to 0.033; uniform is 0.0078) it cancels the percepts' shared part and the loop settles.
# When one stored percept takes the whole attention (peak 1.000 at every stuck body of 415,075),
# the next percept is that row times -2 to -4.5: an overshoot that the size limit of 8 holds in
# place, and each new percept is the anchor's mirror image while the anchor keeps the attention.
# The per-step gain of the loop through one row is at most |eigenvalue| x that row's weight, so a
# peak of 0.05 keeps it at 0.225 or less (4.5 x 0.05). Healthy live recall stayed under 0.05 on
# eight checkpoints (0.033 at most) and is left untouched.
ATTENTION_PEAK_LIMIT=0.05


def cap_attention(scores,limit,iterations=40):
    """softmax(scores / t) per body, with the smallest t >= 1 whose largest weight is at most limit.

    A body with fewer stored percepts than 1/limit cannot reach the limit and gets the uniform
    average. The temperature is found on detached scores (bisection in log t, up to t = e^16) and
    then held constant, so the gradient reaches the scores as through an ordinary softmax.
    """
    with torch.no_grad():
        s=scores.detach().float()
        lo=torch.zeros(len(s),1,device=s.device);hi=torch.full((len(s),1),16.,device=s.device)
        for _ in range(iterations):
            mid=(lo+hi)/2
            too_sharp=(s/mid.exp()).softmax(-1).amax(-1,keepdim=True)>limit
            lo=torch.where(too_sharp,mid,lo);hi=torch.where(too_sharp,hi,mid)
        t=hi.exp()
    if scores.shape[-1]*limit<1:
        return torch.full_like(scores,1./scores.shape[-1])
    return (scores/t.to(scores.dtype)).softmax(-1)


def balanced_function_mean(values,target,valid,initial=None):
    """Equal evidence per body and physical field; retain rare changed sites.

    This is a prediction lesson, never a behavioral reward. Skin quality and
    motor capacity use field zero; exposure and stimulation use fields one and
    two. A numerical zero/one distinguishes exact neutral endpoints only.

    Vectorised 2026-09-18 (S7): the per-body, per-field Python loop it replaces cost
    81.9 ms of every 1,707 ms training step (step_profile_2026-09-17.txt); the value
    and gradient are the loop's to float32 rounding, pinned by
    test_balanced_function_mean_matches_the_loop against _balanced_function_mean_loop.
    """
    if values.shape!=target.shape or valid.shape!=values.shape or not valid.any():
        raise ValueError('missing local function lessons')
    if initial is not None and initial.shape!=target.shape[:1]+target.shape[2:]:
        raise ValueError('initial function layout mismatch')
    B=values.shape[0];F=STATE_FIELDS;eps=8*torch.finfo(target.dtype).eps
    v=values.reshape(B,-1,F).transpose(1,2)             # [B,F,N]: fields first, sites flattened
    t=target.reshape(B,-1,F).transpose(1,2)
    m=valid.reshape(B,-1,F).transpose(1,2)
    endpoint=torch.zeros(F,dtype=t.dtype,device=t.device);endpoint[0]=1.
    affected=(t-endpoint[None,:,None]).abs()>eps          # [B,F,N]
    sel=m[:,:,None,:]&torch.stack([affected,~affected],2) # [B,F,2,N]
    if initial is not None:
        changed=((target-initial[:,None]).abs()>eps).reshape(B,-1,F).transpose(1,2)
        sel=sel[:,:,:,None,:]&torch.stack([changed,~changed],2)[:,:,None,:,:]   # [B,F,2,2,N]
        vv=v[:,:,None,None,:]
    else:
        vv=v[:,:,None,:]
    count=sel.sum(-1)
    mean=(vv*sel).sum(-1)/count.clamp_min(1)             # group means, 0 where a group is empty
    present=count>0
    def _mean_of_present(x,p):                            # mean over the last axis of the groups present
        return (x*p).sum(-1)/p.sum(-1).clamp_min(1),p.any(-1)
    if initial is not None:
        mean,present=_mean_of_present(mean,present)       # over moving / still     -> [B,F,2]
    mean,present=_mean_of_present(mean,present)           # over affected / clear   -> [B,F]
    pairs=m.any(-1)                                       # (body, field) with any valid site
    return (mean*pairs).sum()/pairs.sum()


def _balanced_function_mean_loop(values,target,valid,initial=None):
    """The original per-body, per-field loop, kept as the reference the test compares
    the vectorised form against. Not called by training."""
    if values.shape!=target.shape or valid.shape!=values.shape or not valid.any():
        raise ValueError('missing local function lessons')
    if initial is not None:
        if initial.shape!=target.shape[:1]+target.shape[2:]:raise ValueError('initial function layout mismatch')
        changed=(target-initial[:,None]).abs()>8*torch.finfo(target.dtype).eps
    else:changed=None
    groups=[]
    for body in range(len(values)):
        for field in range(STATE_FIELDS):
            mask=valid[body,...,field]
            if not mask.any():continue
            endpoint=1. if field==0 else 0.
            affected=(target[body,...,field]-endpoint).abs()>8*torch.finfo(target.dtype).eps
            field_groups=[]
            for condition in (affected,~affected):
                group=mask & condition
                if changed is None:
                    if group.any():field_groups.append(values[body,...,field][group].mean())
                else:
                    dynamics=[]
                    for moving in (changed[body,...,field],~changed[body,...,field]):
                        selected=group & moving
                        if selected.any():dynamics.append(values[body,...,field][selected].mean())
                    if dynamics:field_groups.append(torch.stack(dynamics).mean())
            groups.append(torch.stack(field_groups).mean())
    return torch.stack(groups).mean()


class GroundedDynamics(nn.Module):
    """Learn future change from the observed present and executed action history.

    Zero residual means persistence, with explicitly unqualified uncertainty.
    Future outcomes never enter this path. Elapsed physical seconds distinguish
    equal command sequences applied at different cadences.

    change() emits a RATE, in fraction of the 0-to-1 function scale PER SECOND. Each
    horizon multiplies that one output by CUMULATIVE elapsed seconds, so what the head
    is asked for at horizon h is the average rate over the whole window, not a per-step
    increment - a different quantity at every horizon.
    """
    def __init__(self,d_model,action_size):
        super().__init__()
        self.register_buffer('action_center',torch.zeros(action_size))
        self.register_buffer('action_scale',torch.ones(action_size))
        self.initial=nn.Linear(d_model,d_model)
        self.command=nn.Linear(action_size+1,d_model)
        self.history=nn.GRUCell(d_model,d_model)
        self.local=nn.Sequential(nn.Linear(2*d_model+STATE_FIELDS+2,d_model),nn.GELU())
        self.change=nn.Linear(d_model,STATE_FIELDS)
        self.uncertainty=nn.Linear(d_model,STATE_FIELDS)
        self.site_observation=nn.Linear(SITE_INPUT,d_model)
        nn.init.zeros_(self.change.weight);nn.init.zeros_(self.change.bias)
        nn.init.zeros_(self.uncertainty.weight);nn.init.zeros_(self.uncertainty.bias)
        nn.init.zeros_(self.site_observation.weight);nn.init.zeros_(self.site_observation.bias)
        from .spatial_self import SpatialSelfContext
        self.spatial_context=SpatialSelfContext(d_model)
        self.scope_encoder=nn.Linear(2,d_model)
        nn.init.zeros_(self.scope_encoder.weight);nn.init.zeros_(self.scope_encoder.bias)
        # ONE free additive constant on the site token, not two (audit P213). Both
        # encoders add their output into the same tensor x (perceive()), and an
        # nn.Linear bias gradient does not depend on that layer's input - it is the
        # upstream gradient summed over the batch and site axes - so two zero-initialised
        # biases receive the identical gradient every step and can never separate.
        # Measured on H83 ckpt_step_4097248: torch.equal(scope.bias,surface.bias) True
        # after 4,097,248 steps, the only equal pair among the 17 128-wide self_state
        # biases, and the constant they form moved 1.99x per AdamW step what a single
        # bias moves. scope_encoder keeps the constant because
        # self_state_integration.py supplies surface_moments only alongside
        # function_extent, never on its own. This bias is retired at zero and frozen,
        # and stays a registered parameter so the optimizer's group sizes are unchanged
        # and a resumed checkpoint's Adam moments still load
        # (checkpoint.py:_migrate_optimizer_state handles growth only, not removal).
        self.surface_encoder=nn.Linear(20,d_model)
        nn.init.zeros_(self.surface_encoder.weight);nn.init.zeros_(self.surface_encoder.bias)
        self.surface_encoder.bias.requires_grad_(False)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,
                              unexpected_keys,error_msgs):
        """Fold an older lineage's duplicated surface bias into the surviving scope bias.

        A checkpoint written before the retirement carries the same constant twice and the
        site token it produced was the SUM, so adding the two keeps that token unchanged.
        Measured on H83 ckpt_step_4097248 over its own stored 8x53 site inputs: the learned
        body-condition readout changes by at most 1.19e-07 on the 0-to-1 function scale.
        Gated on a nonzero saved bias, so it fires once and every later resume is a no-op.
        """
        legacy=state_dict.get(prefix+'surface_encoder.bias')
        target=prefix+'scope_encoder.bias'
        if legacy is not None and target in state_dict and legacy.count_nonzero():
            state_dict[target]=state_dict[target]+legacy
            state_dict[prefix+'surface_encoder.bias']=torch.zeros_like(legacy)
            log.warning('self_state: folded the retired duplicate surface_encoder.bias '
                        '(absmean %.6g) into scope_encoder.bias; the site token is unchanged',
                        float(legacy.abs().mean()))
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,
                                      unexpected_keys,error_msgs)

    @torch.no_grad()
    def calibrate_actions(self,actions):
        """Training-only input calibration, fixed before reserved worlds are read.
        Used by the offline tool and its test; the training path uses
        recalibrate_actions below, which keeps every predicted value unchanged."""
        if actions.ndim!=2 or actions.shape[1]!=len(self.action_center) or len(actions)<2 or not torch.isfinite(actions).all():
            raise ValueError('action calibration needs finite recorded executed commands')
        center=actions.mean(0);scale=actions.std(0,unbiased=False)
        scale=torch.where(scale>8*torch.finfo(actions.dtype).eps,scale,torch.ones_like(scale))
        self.action_center.copy_(center);self.action_scale.copy_(scale)

    @torch.no_grad()
    def recalibrate_actions(self,actions):
        """Refresh the calibration without changing a single predicted value (audit P203).

        command=(a-c)/s at forward() feeds a Linear, so a new centre and scale are
        exactly absorbed by W'[:,i]=W[:,i]*s'_i/s_i and b'=b+W@((c'-c)/s). Measured
        2026-09-16 on the H83 checkpoint at step 4,097,248 over that run's own 1,024
        stored executed orders: the forecast mean moves by at most 5.96e-08 of the
        0-to-1 function scale (half a float32 rounding step at 1.0) and the variance
        by at most 7.63e-06, while the normalised command falls from absolute mean
        6.4159 (largest 232.8) to 0.7566 (largest 11.1). Before this method existed no
        training step refreshed the two buffers: they held whatever an offline pass
        last wrote (H83: action_scale minimum 0.00322426, multiplying one command
        number by 310).
        """
        if actions.ndim!=2 or actions.shape[1]!=len(self.action_center) or len(actions)<2 or not torch.isfinite(actions).all():
            raise ValueError('action calibration needs finite recorded executed commands')
        before=float(((actions-self.action_center)/self.action_scale).abs().mean())
        center=actions.mean(0);scale=actions.std(0,unbiased=False)
        scale=torch.where(scale>8*torch.finfo(actions.dtype).eps,scale,torch.ones_like(scale))
        n=len(self.action_center)
        factor=scale/self.action_scale
        # Order is load-bearing: the bias update must read the OLD weight columns and
        # the OLD scale, so it comes before the weight rescale and before the copies.
        self.command.bias.add_(self.command.weight[:,:n]@((center-self.action_center)/self.action_scale))
        self.command.weight[:,:n]=self.command.weight[:,:n]*factor
        self.action_center.copy_(center);self.action_scale.copy_(scale)
        after=float(((actions-self.action_center)/self.action_scale).abs().mean())
        return dict(samples=int(len(actions)),normalised_absmean_before=before,
                    normalised_absmean_after=after,scale_factor=factor.detach().cpu())

    def forward(self,percept,actions,dt):
        observed=percept['observed'];local=percept['local']+self.site_observation(percept['observed_sites'])
        if 'spatial' in percept:
            local=local+self.spatial_context(local,percept['spatial'],percept['spatial_valid'])
        h=self.initial(percept['belief']);elapsed=torch.zeros_like(dt[:,:1])
        means=[];variances=[]
        for index in range(actions.shape[1]):
            duration=dt[:,index:index+1];elapsed=elapsed+duration
            command=(actions[:,index]-self.action_center)/self.action_scale
            h=self.history(self.command(torch.cat([command,torch.log1p(duration)],-1)),h)
            clocks=torch.cat([torch.log1p(elapsed),torch.log1p(duration)],-1)
            features=self.local(torch.cat([local,h[:,None].expand(-1,local.shape[1],-1),observed,
                                           clocks[:,None].expand(-1,local.shape[1],-1)],-1))
            # Direct state residual avoids repeatedly integrating an accumulated
            # error. Recurrent h still sees every executed order in sequence.
            # change() is a rate per second; elapsed is cumulative seconds, so the
            # product is a fraction of the 0-to-1 function scale. On the H83 checkpoint
            # at step 4,097,248 the head emits -4.28 to +2.28 per second at horizon 1
            # (absolute mean 0.1125) against a measured actual change of 0.0203 per
            # second on average, 0.337 at the 99th percentile and 18.59 at its largest,
            # over 192,024 compared site-values.
            means.append((observed+elapsed[:,None]*self.change(features)).clamp(0,1))
            variances.append(self.uncertainty(features.detach()).clamp(-9,3).exp())
        return torch.stack(means,1),torch.stack(variances,1)


class SelfHistory:
    """Per-body recurrent state and past relations, checkpointed with physical time."""
    def __init__(self,capacity=128):
        self.capacity=capacity;self.reset()
    def reset(self):
        self.state=None;self.internal=None;self.time=None;self.sites=None;self.valid=None
        self.context=None;self.relations=deque(maxlen=self.capacity)
        self.targets=None
        self.intention=None
        self.site_ids=None
        # Exact causal inputs, including the recurrent predecessor and earlier
        # relations. Re-encoding a prior frame as a fresh episode is not equivalent.
        self.inputs=None
        self.events=deque(maxlen=self.capacity)
    def state_dict(self):
        return map_state(dict(capacity=self.capacity,**{k:getattr(self,k) for k in
            ('state','internal','time','sites','valid','context','intention','site_ids','inputs','targets')},
            relations=list(self.relations),events=list(self.events)), 'cpu')
    def load_state_dict(self,s,device):
        if s['capacity']!=self.capacity:raise ValueError('relation memory budget changed without migration')
        for k in ('state','internal','time','sites','valid','context','intention','site_ids','inputs','targets'):
            setattr(self,k,map_state(s.get(k),device))
        self.relations=deque((x.to(device) for x in s['relations']),maxlen=self.capacity)
        self.events=deque(map_state(s.get('events',[]),device),maxlen=self.capacity)

    def store(self,belief,released=None):
        """Put this step's self-percept [bodies, width] into the relation memory.

        released is None - the default, and the only behaviour before 2026-09-26 - or a
        [bodies] bool mask of the bodies whose live percept perceive() had to rescale
        (--self-percept-release on). For those bodies every stored row and this step's row
        become zero, so the next recall starts from nothing; other bodies' rows are kept.
        Returns the number of stored rows set to zero.

        Why (docs/gain_of_function_v4/agents/self_percept_runaway_2026-09-26.md): the
        rescale in perceive() keeps a percept's direction, so a body whose stored percepts
        sit at BELIEF_LIMIT recalls them 2 to 4 times larger, is rescaled back to the limit
        and stores that again. Re-running H84's recorded steps (checkpoints 407,032 to
        415,075), the loop stayed over the limit on 128 of 128 steps from the recorded
        memory and never crossed it from an empty memory; with this release it crossed on
        the first step only. Zeroing ALL of that body's rows, not only rows at the limit, is
        measured too: dropping only the stored percepts that sat at the limit, and storing
        nothing while over it, left one stored percept that took the whole attention, and
        the body stayed over the limit on 128 of 128 steps.
        """
        if released is None or not bool(released.any()):
            self.relations.append(belief);return 0
        cleared=self.release(released)
        self.relations.append(torch.where(released.to(belief.device).reshape(-1,1),torch.zeros_like(belief),belief))
        return cleared

    def release(self,bodies):
        """Set every stored row of the bodies in the [bodies] bool mask to zero; returns the rows cleared."""
        if not self.relations:return 0
        clear=bodies.to(self.relations[0].device).reshape(-1,1)
        cleared=len(self.relations)*int(bodies.sum())
        self.relations=deque((torch.where(clear,torch.zeros_like(r),r) for r in self.relations),
                             maxlen=self.capacity)
        return cleared


def map_state(value,device):
    """Detach state without aliasing subsequent live observations or saved snapshots."""
    if torch.is_tensor(value):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:map_state(v,device) for k,v in value.items()}
    if isinstance(value,list):return [map_state(v,device) for v in value]
    if isinstance(value,tuple):return tuple(map_state(v,device) for v in value)
    return copy.deepcopy(value)
