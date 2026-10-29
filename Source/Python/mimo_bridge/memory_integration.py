# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Observed bodily needs and the persistent memory inputs belonging to one decision.

A MEMORY EVENT's columns (observe, below), in order: one d_model token per active modality (the
senses), the NEED_CHANNELS measured needs, their NEED_CHANNELS valid flags, the executed order (the
ledger row in encoder units, command_for_encoder: 9 numbers per handle, the quaternion passed through
unchanged), the goal summary (d_model). action_columns and eye_row_columns name the order's columns;
LemLite.convert_stored_eye_rows rewrites the eye row's x and y there at an --eye-row-convention switch
(the motor fixes deploy branch's second review, finding 1), and a test fails if those columns move."""
import torch
from lem.model.contact_wellness import contact_response

NEED_CHANNELS = 5          # pain, pleasant, capacity, pressure, shear (measured_needs)


def action_columns(model):
    """The slice of a memory event's columns that holds the executed order (observe's layout)."""
    start = len(model.active) * model.d_model + 2 * NEED_CHANNELS
    return slice(start, start + len(model.handle_names) * 9)


def eye_row_columns(model):
    """The two columns of a memory event that hold the eye row's quaternion x and y."""
    start = action_columns(model).start + int(model.GAZE_IDX) * 9
    return [start + 3, start + 4]


def measured_needs(extras,batch,device):
    raw=extras.get('raw_tactile')
    if raw is None:
        pain=torch.zeros(batch,1,device=device);pleasant=torch.zeros_like(pain)
    else:
        r=contact_response(raw['features'],raw.get('sensitivity'),raw.get('valid'),
                           profile=raw.get('profile','legacy'),history=raw.get('history'))
        pain,pleasant=r['pain'],r['pleasant']
    cap=extras.get('capability')
    # Follow the actual grounded capability formula's first ten capacity channels.
    if cap is None: capacity=torch.ones(batch,1,device=device)
    else: capacity=cap[:,:10].float().mean(-1,keepdim=True).clamp(0,1)
    # UNITS. Skin feature channels 0 and 1 arrive ALREADY divided: pressure by 100,000 Pa
    # and shear by 50,000 Pa (tokenizer/modalities/touch.py _FIELD_NORMS,
    # mimo_bridge/skin_contract.py; lem/model/contact_wellness.py multiplies channel 0 back
    # by 100000 to recover Pascals). They are not Pascals here. Measured on the H83 run's
    # 137 mimo records, mean active-patch pressure 921.5 to 1389.5 Pa is 0.0092 to 0.0139
    # on this channel, so two of the five need channels occupy 1.4% of the 0-to-1 range
    # the memory's contract states, and the x/(1+x) squash below - written for a quantity
    # of order 1 - changes the value by at most 1.37% in that regime.
    pressure=raw['features'][...,0].amax(1,keepdim=True).clamp_min(0) if raw is not None else torch.zeros_like(pain)
    shear=raw['features'][...,1].amax(1,keepdim=True).clamp_min(0) if raw is not None else torch.zeros_like(pain)
    values=torch.cat([pain,pleasant,capacity,pressure/(1+pressure),shear/(1+shear)],-1).detach()
    valid=torch.tensor([raw is not None,raw is not None,cap is not None,raw is not None,raw is not None],device=device).expand(batch,-1)
    return values,valid


def observe(model,tokens,mask,extras,frames,executed):
    b=len(frames);d=model.device
    needs,valid=measured_needs(extras,b,d)
    model._memory_needs_valid=valid
    senses=torch.cat([(tokens[m].reshape(b,-1).detach().float()
                       if mask.get(m) in ('real','stale') else torch.zeros(b,model.d_model,device=d))
                      for m in model.active],-1)
    from mimo_bridge.lem_model import command_for_encoder
    action=torch.zeros(b,len(model.handle_names)*9,device=d) if executed is None else command_for_encoder(executed.detach().float()).reshape(b,-1)
    # Prior selected goal is the model's own state, not a supplied task instruction.
    goal=getattr(model,'_last_plan_inputs',None)
    goal_tokens=goal.get('goal',{}) if goal else {}
    goal_summary=torch.stack([v.detach().float() for v in goal_tokens.values()]).mean(0) if goal_tokens else torch.zeros(b,model.d_model,device=d)
    features=torch.cat([senses,needs,valid.float(),action,goal_summary],-1)
    now=torch.tensor([f.sim_time for f in frames],device=d,dtype=torch.float32)
    model.memory_history.observe(features,now,model.persistent_memory)
    snapshot=model.memory_history.snapshot(features,now)
    if model.persistent_memory.repair!='off' and not model.persistent_memory.repair_switched_py:
        # --persistent-memory-repair: the function-preserving switch (lem/model/persistent_memory.py,
        # LemLite._persistent_memory_switch). LemLite.step switches at its start when the history holds a
        # lived moment; this is the first moment of an empty history, where nothing earlier in the step
        # read the memory (the delayed lesson's _nf_memory is None then), so the fold's in-place change of
        # the readout weight cannot meet a graph that holds it.
        model._persistent_memory_switch(snapshot)
    # observe=True: the one lived call of the step (the repair's running means; ignored without it)
    state,_=model.persistent_memory.encode(**snapshot,observe=True)
    return state,model.persistent_memory.bias(state,site="lived"),snapshot,needs
