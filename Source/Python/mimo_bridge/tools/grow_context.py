# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Grow direct attention while preserving recency and every earned Adam moment."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file, save_file

KEY = 'forward_model.temporal_embedding.weight'


def grow_table(w, n_new, *, moments=False):
    """Sequences are oldest first. Keep every old row at its original recency."""
    n_old = w.shape[0]
    if n_new < n_old:
        raise ValueError('context shrinking would discard earned temporal positions')
    if n_new == n_old:
        return w.clone()
    out = torch.zeros((n_new, *w.shape[1:]), dtype=w.dtype, device=w.device)
    out[-n_old:] = w
    if not moments:
        # New older positions begin at the oldest known embedding. Distinction
        # must be learned; an invented trend is not a measured age embedding.
        out[:n_new-n_old] = w[0]
    return out


def grow_checkpoint(src, n_new, dst):
    from mimo_bridge.tools.causal_test import _build_model
    from mimo_bridge.tools.migrate_trunk import modules
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if dst.exists(): raise FileExistsError(dst)
    old, args, _, found = _build_model(str(src), 'cpu', 0, intention=True, load=False)
    if not found: raise ValueError('a located run configuration is required')
    saved = load_file(str(src/'model.safetensors'))
    n_old = saved[KEY].shape[0]
    if n_new <= n_old: raise ValueError('new context must be larger')
    new, _, _, _ = _build_model(str(src), 'cpu', 0, intention=True,
                                arg_overrides={'context_length': n_new}, load=False)
    templates = {f'{k}.{n}': v.cpu() for k, mod in modules(new).items() for n,v in mod.state_dict().items()}
    if set(saved) != set(templates):
        raise ValueError(f'context migration cannot change module names: {set(saved)^set(templates)}')
    grown, changed = {}, []
    for name, value in saved.items():
        if name == KEY: grown[name] = grow_table(value, n_new)
        elif name.endswith('.causal_mask'):
            if value.dtype != torch.bool or value.shape != (n_old,n_old):
                raise ValueError(f'unexpected causal mask {name}')
            grown[name] = templates[name]
        elif value.shape == templates[name].shape: grown[name] = value.clone()
        else: raise ValueError(f'unreviewed shape change: {name}')
        if grown[name].shape != value.shape: changed.append(name)
    old_names = {id(p): f'{k}.{n}' for k,m in modules(old).items() for n,p in m.named_parameters()}
    new_names = {id(p): f'{k}.{n}' for k,m in modules(new).items() for n,p in m.named_parameters()}
    packed = torch.load(src/'optimizer.pt', map_location='cpu', weights_only=False)
    result = copy.deepcopy(packed); opt = result['optimizer']; count = 0
    if len(opt['param_groups']) != len(old.optimizer.param_groups) or len(opt['param_groups']) != len(new.optimizer.param_groups):
        raise ValueError('optimizer groups changed')
    for stored, og, ng in zip(opt['param_groups'], old.optimizer.param_groups, new.optimizer.param_groups):
        names = [old_names[id(p)] for p in og['params']]
        if names != [new_names[id(p)] for p in ng['params']] or len(names) != len(stored['params']):
            raise ValueError('optimizer ordering changed')
        for pid, name in zip(stored['params'], names):
            state = opt['state'].get(pid, {}); count += bool(state)
            if name == KEY:
                for k,v in state.items():
                    if torch.is_tensor(v) and v.ndim:
                        if v.shape != saved[KEY].shape: raise ValueError(f'wrong optimizer moment shape {k}')
                        state[k] = grow_table(v, n_new, moments=True)
    dst.mkdir(parents=True)
    for p in src.iterdir():
        if p.is_file() and p.name not in ('model.safetensors','optimizer.pt','model_receipt.json'):
            shutil.copy2(p,dst/p.name)
    save_file(grown,str(dst/'model.safetensors')); torch.save(result,dst/'optimizer.pt')
    (dst/'model_receipt.json').write_text(json.dumps(new.receipt,indent=2),encoding='utf-8')
    (dst.parent.parent/'args.json').write_text(json.dumps(dict(args,context_length=n_new),indent=2),encoding='utf-8')
    loaded=new.load_checkpoint(str(dst),load_optimizer=True)
    if loaded.get('reinitialized_components'): raise RuntimeError(f'migration reinitialized components: {loaded}')
    actual={f'{k}.{n}':v.cpu() for k,m in modules(new).items() for n,v in m.state_dict().items()}
    for k,v in grown.items():
        if not torch.equal(v,actual[k]): raise RuntimeError(f'loaded tensor mismatch: {k}')
    after=new.optimizer.state_dict()
    for pid,st in opt['state'].items():
        for k,v in st.items():
            a=after['state'][pid][k]
            if torch.is_tensor(v):
                if not torch.equal(v,a.cpu()): raise RuntimeError(f'optimizer mismatch: {pid}/{k}')
            elif a != v: raise RuntimeError(f'optimizer counter mismatch: {pid}/{k}')
    receipt=dict(parent=str(src),checkpoint=str(dst),old_context=n_old,new_context=n_new,
                 temporal_order='oldest to newest; old rows aligned at recent end',
                 optimizer_states_preserved=count,changed_shapes=changed,
                 old_embedding_rows_exact=True,new_moment_rows_zero=True,
                 all_other_weights_and_optimizer_values_exact=True,
                 runtime_state_copied=True,step=new.step_count)
    (dst/'context_growth.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('ckpt_dir');ap.add_argument('new_context',type=int)
    ap.add_argument('--out');a=ap.parse_args(argv)
    print(json.dumps(grow_checkpoint(a.ckpt_dir,a.new_context,a.out or f'{a.ckpt_dir}_ctx{a.new_context}'),indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
