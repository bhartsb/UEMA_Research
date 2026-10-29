# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Hold the whole model in memory at once and time the replay pass.

The separate sweep measured the core and vision ALONE. A real run keeps the core,
all six encoders and the optimizer state in the same 8 GB, so the replay pass
reaches its thrash point earlier. This measures candidate configurations with
everything allocated together.

ONE DELIBERATE OVER-COUNT, so the numbers are conservative rather than wrong: this
allocates a second ForwardModel as "the planner's frozen copy". The real planner
holds NO copy - intention_generator.py freezes the live weights in place for one
planning call (requires_grad off, prior flags restored on exit). So every peak
reported here includes one extra core the run will not allocate: at 8 blocks x
256 that is 10,571,776 parameters, 42.3 MB in fp32. Subtract it to get the real
peak; it does not change any verdict.
"""
import os
import statistics
import sys

import torch

sys.path.insert(0, 'Source/Python')
from mimo_bridge.lem_model import build_config
from lem.model.forward_model import ForwardModel
from lem.model.encoder_registry import EncoderRegistry

dev = torch.device('cuda:0')
CTX, NMOD, B = 128, 6, 8
STEP_MS = 1000.0 / 0.6494392952843702


def med(fn, n=12, warm=4):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        xs.append(s.elapsed_time(e))
    return statistics.median(xs)


def trial(core_blocks, d, vis_blocks, vis_d):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = build_config({'model': {'d_model': d, 'context_length': CTX,
                                  'forward_model': {'n_layers': core_blocks}},
                        'modalities': {'vision': {'encoder': {
                            'n_swin_layers': vis_blocks, 'd_internal': vis_d,
                            'angular_pos_hidden': vis_d}}}})
    core = ForwardModel(cfg).to(dev)
    # NOT what the real planner does - it freezes the live core in place, no copy.
    # Kept as a deliberate over-count (42.3 MB at 8x256) so the peak is conservative.
    planner_copy = ForwardModel(cfg).to(dev).eval()
    for p in planner_copy.parameters():
        p.requires_grad_(False)
    reg = EncoderRegistry(cfg).to(dev)                         # all six encoders
    params = list(core.parameters()) + list(reg.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4)                   # optimizer moments = 2x params
    # materialise Adam state so its memory is actually held
    loss = sum(p.float().sum() * 0 for p in params)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)

    n_core = sum(p.numel() for p in core.parameters())
    n_enc = sum(p.numel() for p in reg.parameters())

    def replay():
        x = torch.randn(48, CTX * NMOD, d, device=dev, requires_grad=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = x
            for b in core.blocks:
                h = b(h)
        h.float().square().mean().backward()

    def live():
        x = torch.randn(B, CTX * NMOD, d, device=dev, requires_grad=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = x
            for b in core.blocks:
                h = b(h)
        h.float().square().mean().backward()

    try:
        r = med(replay); l = med(live)
        peak = torch.cuda.max_memory_allocated() / 2**20
        status = 'ok'
    except torch.cuda.OutOfMemoryError:
        r = l = float('nan'); peak = torch.cuda.max_memory_allocated() / 2**20; status = 'OUT OF MEMORY'
    del core, planner_copy, reg, opt
    torch.cuda.empty_cache()
    return dict(core=f'{core_blocks}x{d}', vision=f'{vis_blocks}x{vis_d}', core_params=n_core,
                enc_params=n_enc, live_ms=l, replay48_ms=r, peak_mib=peak, status=status)


print(f'{"core":>8} {"vision":>8} {"core par":>11} {"enc par":>11} {"live":>7} {"replay48":>9} {"peak MiB":>9}  status')
base = None
for cb, d, vb, vd in [(2, 128, 2, 128),       # today
                      (8, 128, 4, 128),       # the full design
                      (8, 256, 4, 256),
                      (8, 256, 8, 256),
                      (8, 256, 12, 256),
                      (12, 256, 8, 256)]:
    res = trial(cb, d, vb, vd)
    print(f'{res["core"]:>8} {res["vision"]:>8} {res["core_params"]:11,d} {res["enc_params"]:11,d} '
          f'{res["live_ms"]:7.1f} {res["replay48_ms"]:9.1f} {res["peak_mib"]:9,.0f}  {res["status"]}')
