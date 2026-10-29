"""Liveness sweep of every learned part on real lived moments (read-only diagnosis).
For each Linear followed by GELU/ReLU: share of hidden units whose input is ever above -3 (live).
For each MultiheadAttention: mean entropy of its attention rows against the uniform maximum."""
import os, sys, json, math
if __name__ == "__main__":  # only when run as a program: importing never changes the devices (2026-09-26)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, torch, torch.nn as nn
from mimo_bridge.tools import gof_probe_harness as H
CK = sys.argv[1]
_orig_lc = None
if os.environ.get("ALLOW_TACTILE_REINIT"):
    import mimo_bridge.lem_model as LM
    _orig_lc = LM.LemLite.load_checkpoint
    def _lc(self, *a, **k):
        r = _orig_lc(self, *a, **k)
        re_ = list(r.get("reinitialized_components") or [])
        extra = [c for c in re_ if c != "encoder:tactile"]
        if extra:
            raise RuntimeError("reinitialised beyond the tactile encoder: %s" % extra)
        print("NOTE reinitialised (allowed for dating only):", re_)
        r = dict(r); r["reinitialized_components"] = []
        return r
    LM.LemLite.load_checkpoint = _lc
fr = H.load_frozen(CK, threads=4)
m = fr.model
acts = {}; atts = {}
hooks = []
def reg_seq(name, seq):
    mods = list(seq.children())
    for i in range(len(mods) - 1):
        if isinstance(mods[i], nn.Linear) and isinstance(mods[i + 1], (nn.GELU, nn.ReLU)):
            key = f"{name}.{i}"
            def hk(mod, inp, out, key=key):
                o = out.detach().float().reshape(-1, out.shape[-1])
                a = acts.setdefault(key, [None, None, 0])
                mx = o.max(0).values
                a[0] = mx if a[0] is None else torch.maximum(a[0], mx)
                a[2] += o.shape[0]
            hooks.append(mods[i].register_forward_hook(hk))
ALL = []
for attr, val in vars(m).items():
    if isinstance(val, nn.Module):
        for sub, mod in val.named_modules():
            ALL.append((attr + ('.' + sub if sub else ''), mod))
for name, mod in ALL:
    if isinstance(mod, nn.Sequential):
        reg_seq(name, mod)
    if isinstance(mod, nn.MultiheadAttention):
        orig = mod.forward
        def fwd(*a, _orig=orig, _name=name, **k):
            k["need_weights"] = True; k["average_attn_weights"] = False
            out, w = _orig(*a, **k)
            if w is not None:
                w = w.detach().float()
                n = w.shape[-1]
                ent = -(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum(-1)
                rec = atts.setdefault(_name, [0.0, 0, n, 0.0])
                rec[0] += float(ent.sum()); rec[1] += ent.numel(); rec[2] = n
                rec[3] = max(rec[3], float(w.max(-1).values.mean()))
            return out, w
        mod.forward = fwd
n = 0
for s, snap in H.start_states(fr, 4):
    for _ in range(6):
        s.observe(); s.policy_act(); n += 1
report = {"moments": n, "dead_layers": {}, "live_layers": {}, "attention": {}}
for k, (mx, _, cnt) in acts.items():
    live = float((mx > -3.0).float().mean())
    (report["dead_layers"] if live < 0.5 else report["live_layers"])[k] = round(live, 3)
for k, (se, c, nkeys, pmax) in atts.items():
    if c:
        report["attention"][k] = {"entropy_share_of_uniform": round((se / c) / math.log(max(nkeys, 2)), 3),
                                  "keys": nkeys, "mean_max_weight": round(pmax, 3)}
print(json.dumps(report, indent=1))
