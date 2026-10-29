# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Activity-conditioned re-slice of saved tenure-probe events (2026-09-04).

The probe's per-sense slice advantages are means over every event in the
probe's own 1800-step episode, so quiet episodes read low and active ones
read high (four probes: gate open 77%/58%/33% gave vision 0.010/0.006/-0.000).
This tool re-reads the SAME saved events files and reports each sense's
advantage twice: over gate-open events only, and over gate-closed events,
beside the raw mean - making probe-to-probe comparisons activity-comparable.
No collection re-run; applies retroactively to every saved events file.

Usage: python -m mimo_bridge.tools.probe_reslice <events.pt> [<events.pt> ...]
"""
from __future__ import annotations

import json
import sys

import torch
import torch.nn.functional as F


def reslice(path: str) -> dict:
    from mimo_bridge.lem_model import command_for_encoder
    from mimo_bridge.tools.tenure_probe_v2 import _delta

    blob = torch.load(path, weights_only=False)
    events, meta = blob["events"], blob["meta"]
    d = meta["d_model"]
    proj = torch.nn.Sequential(
        torch.nn.Linear(blob["proj_in"], 256), torch.nn.GELU(),
        torch.nn.Linear(256, 3 * d))
    proj.load_state_dict(blob["order_proj"])
    proj.eval()

    B = events[0]["order"].shape[0]
    with torch.no_grad():
        qs = [proj(command_for_encoder(ev["order"]).reshape(B, -1)) for ev in events]
        deltas, hvs = zip(*[_delta(ev, d) for ev in events])

    HV = torch.cat(hvs).bool()
    out = {"events_file": path, "n_rows": int(HV.numel()),
           "gate_open_frac": float(HV.float().mean())}

    def cos(a, b):
        return F.cosine_similarity(a, b, dim=-1)

    for si, name in ((0, "vision"), (1, "tactile"), (2, "proprioception")):
        sl = slice(si * d, (si + 1) * d)
        a_t = cos(torch.cat(qs)[:, sl], torch.cat(deltas)[:, sl])
        rolls = []
        for k in range(len(events)):
            q, delta = qs[k], deltas[k]
            rd = torch.stack([cos(q[:, sl], torch.roll(delta, s, 0)[:, sl])
                              for s in range(1, B)]).mean(0)
            rolls.append(rd)
        adv = a_t - torch.cat(rolls)          # same-instant-decoy advantage per row
        row = {"raw_mean": float(adv.mean())}
        for label, mask in (("gate_open", HV), ("gate_closed", ~HV)):
            n = int(mask.sum())
            row[label] = {"mean": float(adv[mask].mean()) if n else None, "n": n}
        out[name] = row
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for p in sys.argv[1:]:
        print(json.dumps(reslice(p), indent=1))
