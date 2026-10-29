"""Why the tenure pose head is dead and why orders barely move the motor token (diagnosis, read-only)."""
import os, sys, json, math
if __name__ == "__main__":  # only when run as a program: importing never changes the devices (2026-09-26)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, torch
from mimo_bridge.tools import gof_probe_harness as H
from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, command_for_encoder
CK = sys.argv[1]
fr = H.load_frozen(CK, threads=4)
m = fr.model
out = {"checkpoint": CK}
moms = []
for s, snap in H.start_states(fr, 8):
    s.restore(snap)
    moms.append(s.moment())
    if len(moms) >= 8: break
print("moments", len(moms), type(moms[0]))
X = []; Z = []
head = m.tenure_heads["pose"]
for mom in moms:
    tokens = {k: v.reshape(1, 1, m.d_model) for k, v in mom.tokens.items()}
    kw = {} if mom.mem is None else {"memory_bias": mom.mem}
    with torch.no_grad():
        _l, last, _c = m.forward_model(tokens, mom.ctx, **kw)
    X.append(last.reshape(-1).float())
X = torch.stack(X)                      # [n, 1152]
W = head[0].weight.float(); b = head[0].bias.float()
pre = X @ W.T + b
xm = X.mean(0); dev = X - xm
out["latent_norm"] = float(X.norm(dim=1).mean())
out["latent_common_norm"] = float(xm.norm())
out["latent_varying_norm"] = float(dev.norm(dim=1).mean())
out["pre_min"], out["pre_max"] = float(pre.min()), float(pre.max())
out["pre_from_common_mean"] = float((W @ xm).mean())
out["pre_from_varying_absmean"] = float((dev @ W.T).abs().mean())
u = xm / xm.norm()
cos_rows = (W @ u) / W.norm(dim=1)
out["rows_cos_with_common_dir_mean"] = float(cos_rows.mean()); out["rows_cos_min"] = float(cos_rows.min()); out["rows_cos_max"] = float(cos_rows.max())
# which slots carry the common part: per modality token norm of the mean latent
per = xm.reshape(-1, m.d_model).norm(dim=1)
out["common_norm_per_token"] = {mod: float(v) for mod, v in zip(list(m.active), per)}
# ---- motor encoder: pool attention per handle, and sensitivity per handle's order columns
enc = m.encoder_registry.encoders["motor"]
names = list(getattr(m, "handle_names", [])); out["handles"] = names
att_all = []; sens_cmd = []; sens_ach = []
for mom in moms[:4]:
    ach = mom.ach[..., :MOTOR_ACHIEVED_F].float()
    order = torch.zeros(1, ach.shape[1], 9); order[..., 6] = 1.0; order[..., 8] = 0.5
    cmd = command_for_encoder(order)
    row = torch.cat([ach, cmd], dim=-1).requires_grad_(True)
    x = enc.backbone_forward(row * 1.0)
    q = enc.pool_norm(enc.pool_query.expand(1, -1, -1))
    _o, w = enc.pool_attn(q, x, x, need_weights=True, average_attn_weights=True)
    att_all.append(w.reshape(-1).detach())
    zt = enc.head_forward(x).reshape(-1)
    J = []
    for k in range(zt.numel()):
        g = torch.autograd.grad(zt[k], row, retain_graph=True)[0][0]
        J.append(g)
    J = torch.stack(J)                     # [d, n_handles, 19]
    sens_cmd.append(J[..., MOTOR_ACHIEVED_F:].pow(2).sum(dim=(0, 2)).sqrt())
    sens_ach.append(J[..., :MOTOR_ACHIEVED_F].pow(2).sum(dim=(0, 2)).sqrt())
att = torch.stack(att_all).mean(0); sc = torch.stack(sens_cmd).mean(0); sa = torch.stack(sens_ach).mean(0)
out["pool_attention_per_handle"] = {n: round(float(a), 4) for n, a in zip(names, att)}
out["jacobian_norm_order_columns_per_handle"] = {n: round(float(a), 5) for n, a in zip(names, sc)}
out["jacobian_norm_achieved_columns_per_handle"] = {n: round(float(a), 5) for n, a in zip(names, sa)}
print(json.dumps(out, indent=1))
