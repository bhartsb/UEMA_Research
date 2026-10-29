# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Preserve the current learner while importing only a new dynamics extension."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch


def transplant_grounded(model,candidate_checkpoint):
    """Initialize continuing learning, never issue a physical-choice qualification.

    The source's older body, brain, self encoder, clocks and optimizer histories
    cannot replace the current learner. Newly added neutral spatial inputs can
    remain neutral when the candidate predates them. Already earned destination
    dynamics cannot be overwritten by this initializer.
    """
    from safetensors.torch import load_file
    from lem.training.checkpoint import _migrate_optimizer_state
    from .tools.migrate_trunk import modules
    path=Path(candidate_checkpoint)
    if model.self_state is None or model.self_state.prediction_mode!='grounded_residual':
        raise ValueError('candidate requires the additive grounded learner')
    destination=model.self_state.grounded
    if any(bool(model.optimizer.state.get(param)) for param in destination.parameters()):
        raise ValueError('candidate initializer cannot overwrite earned destination dynamics')
    receipt=json.loads((path/'model_receipt.json').read_text())
    if receipt.get('self_prediction')!='grounded_residual':raise ValueError('source has no grounded dynamics')
    if receipt.get('body_layout','legacy')!=model.body_layout:raise ValueError('candidate body action layout differs')
    weights=load_file(str(path/'model.safetensors'),device='cpu')
    prefix='self_state.grounded.'
    source={key[len(prefix):]:value for key,value in weights.items() if key.startswith(prefix)}
    current=destination.state_dict()
    if not source or set(source)-set(current):raise ValueError('candidate contains unknown grounded parameters')
    for name,value in source.items():
        if value.shape!=current[name].shape or not torch.isfinite(value).all():
            raise ValueError('candidate grounded tensor has incompatible shape or values: '+name)
    missing=set(current)-set(source)
    neutral_prefixes=('spatial_context.','site_observation.','scope_encoder.','surface_encoder.')
    if any(not name.startswith(neutral_prefixes) for name in missing):
        raise ValueError('candidate lacks required dynamics or input calibration')
    # New input projections must have zero output; random keys/queries may be
    # present but their output map stays zero, preserving candidate predictions.
    if any(name.startswith('site_observation.') and current[name].count_nonzero() for name in missing):
        raise ValueError('missing site observation path is no longer neutral')
    if any(name.startswith(('scope_encoder.','surface_encoder.')) and current[name].count_nonzero() for name in missing):
        raise ValueError('missing local extent path is no longer neutral')
    if missing and any(name.startswith('spatial_context.') for name in missing):
        spatial=getattr(destination,'spatial_context',None)
        outputs=[value for name,value in current.items() if name.startswith('spatial_context.') and ('output' in name or 'out.' in name)]
        if spatial is None or not outputs or any(value.count_nonzero() for value in outputs):
            raise ValueError('missing spatial context output is not neutral')
    named={}
    for name,module in modules(model).items():
        for key,param in module.named_parameters():named[id(param)]=name+'.'+key
    saved=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=False)['optimizer']
    mapped,fresh=_migrate_optimizer_state(model.optimizer,saved,named,set(weights))
    current_optimizer=model.optimizer.state_dict();pending=[]
    for group,indices in zip(model.optimizer.param_groups,current_optimizer['param_groups']):
        for param,index in zip(group['params'],indices['params']):
            if named[id(param)].startswith(prefix) and index in mapped['state']:
                state=deepcopy(mapped['state'][index])
                for key,value in state.items():
                    if torch.is_tensor(value):
                        if key!='step' and value.shape!=param.shape:raise ValueError('candidate Adam shape mismatch')
                        if not torch.isfinite(value).all():raise ValueError('nonfinite candidate Adam evidence')
                        state[key]=value.to(param.device) if key!='step' or param.device.type=='cuda' else value
                pending.append((param,state,named[id(param)]))
    # All validation precedes mutations. No original module or optimizer entry is
    # assigned, and the physical continuation is outside this function entirely.
    destination.load_state_dict(source,strict=False)
    for param,state,_ in pending:model.optimizer.state[param]=state
    return dict(schema='grounded_candidate_initialization_v1',source=str(path.resolve()),
                source_weights_sha256=hashlib.sha256((path/'model.safetensors').read_bytes()).hexdigest(),
                copied_tensors=sorted(source),copied_optimizer_states=[name for _,_,name in pending],
                missing_neutral_extensions=sorted(missing),older_brain_imported=False,production_choice_license=False)
