"""M9 (reviews/dashboards.md): the motor encoder's per-handle readability on ckpt_step_344400.

1. Writes the lived-rows fixture docs/gain_of_function_v3/measurements/motor_rows_seg31_ckpt_step_344400.npz
   from b5-synergy's collect_m2.npz (ckpt_step_344400, segment 31's 8-body world, CPU): 16
   (moment, body) pairs spread over time and bodies, the achieved handle state and the order sent.
2. Loads ONLY the motor encoder from the checkpoint (dashboard.load_motor_encoder) and measures, per
   handle, how far a 20-degree turn of that one handle's order moves the token (probes.
   per_handle_readability), and the encoder's pool attention per handle on the same rows (the
   independent cross-check: a handle the pool gives no weight should not move the token).
Writes docs/gain_of_function_v3/measurements/per_handle_readability_ckpt_step_344400.json."""
import json
import os
import sys
import time

if __name__ == "__main__":  # only when run as a program: importing never changes the devices (2026-09-26)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO = r"D:\UEMA_Research"
sys.path.insert(0, os.path.join(REPO, "Source", "Python"))
import numpy as np
import torch

torch.set_num_threads(4)
from mimo_bridge.tools import dashboard as D
from mimo_bridge.tools.probes import per_handle_readability
from mimo_bridge.lem_model import MOTOR_ACHIEVED_F, command_for_encoder

SRC = os.path.join(os.environ["TEMP"], "the builder", "c--Users-bhart-Documents-UEMA-devA-private",
                   "4b10a9e0-5154-445e-820d-d142536fab24", "scratchpad", "b5", "collect_m2.npz")
RUN = os.path.join(REPO, "Saved", "mimo_bridge", "experiments", "H84_fresh_drivers_20260918_seg31_toy_and_push")
CK = os.path.join(RUN, "checkpoints", "ckpt_step_344400")
MEAS = os.path.join(REPO, "docs", "gain_of_function_v3", "measurements")

d = np.load(SRC, allow_pickle=False)
names_rec = [str(x) for x in d["handle_names"]]
pairs = [(t, b) for t, b in zip(range(30, 600, 36), [0, 3, 6, 1, 4, 7, 2, 5] * 3)][:16]
ach = np.stack([d["ach"][t, b] for t, b in pairs]).astype(np.float32)        # [16, 26, 10]
order = np.stack([d["mfa"][t, b] for t, b in pairs]).astype(np.float32)      # [16, 26, 9]
t0 = time.time()
encoder, layout, names = D.load_motor_encoder(os.path.join(CK, "model.safetensors"), RUN)
assert names == names_rec, (names, names_rec)
np.savez(D.READABILITY_ROWS, ach=ach, order=order, layout=np.array(layout), names=np.array(names),
         moments=np.array([t for t, _ in pairs]), bodies=np.array([b for _, b in pairs]),
         source=np.array("b5-synergy collect_m2.npz: ckpt_step_344400, segment 31's 8-body world, "
                         "frozen weights, the checkpoint's own policy (2026-09-22)"))
rows = D.readability_rows(layout, len(names), limit=16)
res = per_handle_readability(encoder, rows, axes=3)

# the pool's attention per handle on the same rows (the cross-check)
enc = encoder.encoders["motor"]
att = []
with torch.no_grad():
    for a, o in rows:
        x = torch.cat([a[..., :MOTOR_ACHIEVED_F], command_for_encoder(o)], dim=-1)
        h = enc.backbone_forward(x)
        q = enc.pool_norm(enc.pool_query.expand(1, -1, -1))
        _o, w = enc.pool_attn(q, h, h, need_weights=True, average_attn_weights=False)
        att.append(w.reshape(-1, len(names)))           # [heads(queries), n]
att = torch.stack(att).mean(0)                          # [heads, n]
per_head_top = [names[int(i)] for i in att.argmax(dim=1)]
att_any = att.max(dim=0).values                         # the largest weight any head gives a handle
share = res["share"]
order_idx = sorted(range(len(names)), key=lambda i: share[i])
srt = [share[i] for i in order_idx]
gaps = [(srt[i + 1] / max(srt[i], 1e-30), i) for i in range(len(srt) - 1)]
ratio, cut = max(gaps)
low_top, high_bottom = srt[cut], srt[cut + 1]
line = float(np.sqrt(low_top * high_bottom))
below = [names[i] for i in order_idx[:cut + 1]]
out = {
    "measured": time.strftime("%Y-%m-%d %H:%M"), "checkpoint": CK, "rows": res["rows"],
    "rows_source": str(d.files and "b5-synergy collect_m2.npz (see the fixture's 'source')"),
    "method": ("motor encoder only, loaded from model.safetensors at the run's size; 16 lived (moment, "
               "body) rows; for each handle alone, its ordered orientation turned 20 degrees about 3 "
               "random world axes, everything else held; token distance from the unturned order, "
               "averaged, divided by the token's own size (mean norm)"),
    "token_size": res["token_size"], "all_at_once_share": res["all_at_once"] / res["token_size"],
    "per_handle_share": {n: s for n, s in zip(names, share)},
    "pool_attention_largest_weight_per_handle": {n: round(float(v), 6) for n, v in zip(names, att_any)},
    "pool_heads_top_handle": per_head_top,
    "largest_gap": {"between": [low_top, high_bottom], "ratio": ratio,
                    "handles_below": below, "count_below": len(below)},
    "line": line,
    "line_rule": ("the geometric middle of the largest gap between neighbouring per-handle shares; "
                  "every handle sits at least sqrt(ratio) times away from it"),
    "seconds": round(time.time() - t0, 1),
}
path = os.path.join(MEAS, "per_handle_readability_ckpt_step_344400.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("token_size", "all_at_once_share", "largest_gap", "line", "pool_heads_top_handle", "seconds")}, indent=1))
for i in order_idx:
    print("%-12s share %.3e  pool max weight %.2e" % (names[i], share[i], float(att_any[i])))
