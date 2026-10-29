# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Measure dense attention cost for the actual checkpoint architecture on this GPU."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import torch
from mimo_bridge.tools.causal_test import _build_model
from mimo_bridge.tools.grow_context import grow_table
from safetensors.torch import load_file


def run(checkpoint,out):
    torch.set_num_threads(2)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    saved=load_file(str(Path(checkpoint)/'model.safetensors'))
    rows=[]
    for context in (64,96,128,192):
        m,_,_,_= _build_model(checkpoint,'cpu',0,intention=True,load=False,arg_overrides={'context_length':context})
        fm=m.forward_model
        weights={k.removeprefix('forward_model.'):v for k,v in saved.items() if k.startswith('forward_model.')}
        weights['temporal_embedding.weight']=grow_table(weights['temporal_embedding.weight'],context)
        for name,v in fm.state_dict().items():
            if name.endswith('causal_mask'): weights[name]=v
        fm.load_state_dict(weights,strict=True);fm.to(device)
        row=dict(context=context,direct_span_seconds=(context-1)*.05,measurements=[])
        try:
            for batch in (8,48):
                tokens=torch.randn(batch,fm._n_modalities,fm._d_model,device=device)
                ctx=torch.randn(batch,(context-1)*fm._n_modalities,fm._d_model,device=device)
                durations=[]
                if device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats();free_before,total=torch.cuda.mem_get_info()
                for rep in range(4):
                    fm.zero_grad(set_to_none=True)
                    if device.startswith('cuda'): torch.cuda.synchronize()
                    start=time.perf_counter()
                    with torch.autocast(device_type='cuda',enabled=device.startswith('cuda'),dtype=torch.bfloat16):
                        seq,last,_=fm(tokens,ctx);loss=last.square().mean()
                    loss.backward()
                    if device.startswith('cuda'): torch.cuda.synchronize()
                    if rep: durations.append(time.perf_counter()-start)
                    del seq,last,loss
                row['measurements'].append(dict(batch=batch,median_forward_backward_ms=statistics.median(durations)*1000,
                    peak_process_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
                    gpu_free_before_bytes=free_before if device.startswith('cuda') else None))
            row['status']='completed'
        except torch.OutOfMemoryError:
            row['status']='out_of_memory'
        rows.append(row)
        del m,fm,tokens,ctx;gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()
    result=dict(device=device,rows=rows,scope='Forward/backward transformer only; shape-matched synthetic tokens; production training remains concurrent. Not complete training throughput or prediction accuracy.')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('checkpoint');ap.add_argument('--out',required=True);a=ap.parse_args();run(a.checkpoint,a.out)
