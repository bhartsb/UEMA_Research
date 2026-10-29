# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Build one snapshot from a stepped LemLite - the capture half of the explainer.

The contract (see __init__.py and the plan's gap register): the attention re-forward
uses model.last_tokens_fed against model.last_context - the exact inputs the live
forward consumed - and the header labels whether the weights have moved since that
pass ("post_update" in training, "frozen" in --eval-only probes).
"""
from __future__ import annotations

import base64
import io
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from lem.model.body_sets import HANDLE_NAMES
from mimo_bridge.explainer.schema import SCHEMA_VERSION, quantize

PATCH = 16  # tokenizer patch size; vision_adapter.PATCH

# THE TACTILE ROW'S COLUMNS, WRITTEN BESIDE THE ROWS. The viewer reads THIS to know which
# column is which, so the page can never be drawing one column while the model fills
# another: each comma-separated entry is one column, except an entry ending in "xN"
# ("vib x4", "push direction x3"), which is N. The entries must add up to
# lem_model.TOUCH_TOKEN_F and the last one must land on lem_model.TOUCH_VALID_COL; a test
# holds both. The trailing divisor on a label ("depth cm/10") is what the page multiplies
# a stored number by to print it in that label's own unit.
TOUCH_CLOUD_DOC = ("[x,y,z (root-relative /0.5m), pressure, shear, slip, area, "
                   "vib x4, self-touch share, own parts on it /4, "
                   "push direction x3, depth cm/0.1, arrival speed cm/s/2, valid]")


# ----------------------------------------------------------------- small helpers
def _r(x: Any, nd: int = 5) -> Any:
    """Round floats recursively so the JSON stays readable and small."""
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    return x


HSELF_SMALL_GAUGES = ("center_drift", "center_drift_max", "center_step")


def hself_block(stats: Any, scalars) -> Any:
    """The hierarchical self's stats for a snapshot. Everything passes through the snapshot's
    5-decimal rounding EXCEPT the three centre-movement gauges: stats() already cuts them to 4
    significant digits and they read 1e-13 to 1e-6, so 5 decimals printed every one of them as
    0.0 on the page (audit P240). Only these three keys change; every other number in the
    snapshot keeps its rounding."""
    out = scalars(stats)
    for k in HSELF_SMALL_GAUGES:
        if isinstance((stats or {}).get(k), float):
            out[k] = stats[k]
    return out


def _lst(t, nd: int = 5):
    """tensor/array -> rounded nested lists (never NaN: JSON has no spelling for it)."""
    a = np.asarray(t.detach().float().cpu().numpy() if torch.is_tensor(t) else t,
                   dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return _r(a.tolist(), nd)


def _png_b64(img: np.ndarray) -> str:
    """uint8 [H, W, 3] -> base64 PNG (matplotlib is already a bridge dependency)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    buf = io.BytesIO()
    mpimg.imsave(buf, img, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------- eye reassembly
def tile_eye_images(vision_patches: np.ndarray,
                    layout: Sequence[Tuple[str, int]]) -> Dict[str, Dict[str, np.ndarray]]:
    """Invert BinocularPatcher: interleaved patch rows -> per-source per-eye images.

    vision_patches: (2N, 768) float in [0, 1]; layout: [(source_name, size_px), ...]
    in the patcher's order (coarsest first). Within each source block rows alternate
    L0, R0, L1, R1, ... and patches tile the image row-major (image_to_patches in
    vision_adapter.py).
    Returns {source: {left: uint8 [size, size, 3], right: ...}}. Raises ValueError
    when the patch count does not match the layout - the caller turns that into a
    labeled absence, never a wrong picture.
    """
    p = np.asarray(vision_patches, dtype=np.float32)
    expect = sum(2 * (size // PATCH) ** 2 for _, size in layout)
    if p.ndim != 2 or p.shape[0] != expect or p.shape[1] != PATCH * PATCH * 3:
        raise ValueError(f"vision_patches {p.shape} does not match layout {list(layout)} "
                         f"(expected ({expect}, {PATCH * PATCH * 3}))")
    out: Dict[str, Dict[str, np.ndarray]] = {}
    row = 0
    for name, size in layout:
        n = (size // PATCH) ** 2
        block = p[row:row + 2 * n]
        row += 2 * n
        grids = {"left": block[0::2], "right": block[1::2]}
        out[name] = {}
        cols = size // PATCH
        for eye, g in grids.items():
            img = g.reshape(cols, cols, PATCH, PATCH, 3).transpose(0, 2, 1, 3, 4)
            img = img.reshape(size, size, 3)
            out[name][eye] = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    return out


def reassemble_eyes(vision_patches: np.ndarray,
                    layout: Sequence[Tuple[str, int]],
                    max_px: int = 128) -> Dict[str, Dict[str, str]]:
    """tile_eye_images + downsample to <= max_px a side + PNG/base64 encode."""
    tiled = tile_eye_images(vision_patches, layout)
    out: Dict[str, Dict[str, str]] = {}
    for name, eyes in tiled.items():
        out[name] = {}
        for eye, img in eyes.items():
            stride = max(1, img.shape[0] // max_px)
            out[name][eye] = _png_b64(img[::stride, ::stride])
    return out


# ----------------------------------------------------------------- attention capture
def capture_attention_maps(model, body: int = 0, check_invariants: bool = False):
    """One no-grad re-forward with capture_attention='all' for one body.

    Returns (raw_layers, last_step):
      raw_layers: per layer {stage1 [ctx, n_mod, n_mod], stage2 [n_mod, ctx, ctx],
                  bridge [S, S]} as float32 numpy (None where a stage is disabled)
      last_step:  [n_mod, d_model] torch tensor - the re-forward's output for the
                  equality test against the live pass's _lag_last. The live pass adds
                  the persistent memory's bias to the newest tokens AFTER the core's
                  last LayerNorm, so that bias is passed here too; without it the two
                  cannot be equal on a run with --persistent-memory on, and the test
                  that names this equality was building its model with memory off.
    """
    if model.last_tokens_fed is None or model.last_context is None:
        raise RuntimeError("no stepped state to capture: run model.step() first")
    fm = getattr(model.forward_model, "_orig_mod", model.forward_model)
    ctx_len, n_mod = model.ctx_len, model.n_mod
    toks = {m: v[body:body + 1] for m, v in model.last_tokens_fed.items()}
    ctx = model.last_context[body:body + 1]
    mbias = getattr(model, "_memory_bias_current", None)
    _mb = {} if mbias is None else {"memory_bias": mbias[body:body + 1]}
    with torch.no_grad():
        _lat, last_step, _cu, layers = fm(toks, context_buffer=ctx,
                                          capture_attention="all", **_mb)
    raw_layers = []
    for lw in layers:
        cm = lw.get("cross_modal")
        ct = lw.get("temporal")
        br = lw.get("bridge")
        entry = {
            "stage1": (None if cm is None else
                       cm.float().cpu().numpy().reshape(ctx_len, n_mod, n_mod)),
            "stage2": (None if ct is None else
                       ct.float().cpu().numpy().reshape(n_mod, ctx_len, ctx_len)),
            "bridge": (None if br is None else br.float().cpu().numpy()[0]),
        }
        if check_invariants:
            for key, ax in (("stage1", -1), ("stage2", -1), ("bridge", -1)):
                a = entry[key]
                if a is not None:
                    s = a.sum(axis=ax)
                    if not np.allclose(s, 1.0, atol=1e-3):
                        raise AssertionError(f"{key} rows do not sum to 1 "
                                             f"(max err {abs(s - 1).max():.2e})")
            if entry["stage2"] is not None:
                upper = np.triu(entry["stage2"], k=1)
                if not np.allclose(upper, 0.0, atol=1e-6):
                    raise AssertionError("stage2 violates the causal mask")
        raw_layers.append(entry)
    return raw_layers, last_step[0]


def _bridge_aggregates(bridge: np.ndarray, ctx_len: int, n_mod: int) -> Dict[str, Any]:
    """[S, S] bridge matrix -> the compact pieces the schema stores (v1 defers the
    full matrix; see the plan's deferred list)."""
    S = ctx_len * n_mod
    a4 = bridge.reshape(ctx_len, n_mod, ctx_len, n_mod)
    rows = bridge[S - n_mod:, :]                      # last frame's query rows [n_mod, S]
    pair = a4.mean(axis=(0, 2))                       # [n_mod, n_mod]
    tm = a4.mean(axis=(1, 3))                         # [ctx, ctx] time-time
    # STAGE 3 IS UNMASKED. Unlike stage 2, the bridge passes no causal mask
    # (forward_model.py), so it attends FORWARD in time as freely as backward.
    # `lag` only ever sampled k >= 0 - the backward half - and the page plotted it
    # under "attention vs how far back" without saying what it left out, so roughly
    # half the attention budget was invisible. Both halves are now reported.
    lag = [float(np.mean([tm[i, i - k] for i in range(k, ctx_len)]))
           for k in range(ctx_len)]
    lag_forward = [float(np.mean([tm[i, i + k] for i in range(ctx_len - k)]))
                   for k in range(ctx_len)]
    # Neither curve is a SHARE: each entry averages a different number of cells, so
    # summing them answers nothing. This is the share - three numbers over the same
    # total that add to 1 - and it is what "how much looks back" actually means.
    total = float(tm.sum())
    if total > 0:
        rows_i, cols_j = np.indices(tm.shape)
        direction_share = {
            "backward": float(tm[cols_j < rows_i].sum() / total),
            "same_frame": float(np.trace(tm) / total),
            "forward": float(tm[cols_j > rows_i].sum() / total),
        }
    else:
        direction_share = {"backward": None, "same_frame": None, "forward": None}
    return {"rows": rows, "pair": pair, "lag": lag,
            "lag_forward": lag_forward, "direction_share": direction_share}


# ----------------------------------------------------------------- schema v3 (2026-09-22)
# THE GAIN-OF-FUNCTION BLOCKS. Every one is read from an attribute the model already holds after
# its step - never an extra forward pass, and never a *_window_summary(), which RESETS the stats
# window it summarises (calling one here would blank the next stats record). A feature that is
# off gives a labelled absence, never a missing key the page must guess about.
def _plain(x: Any, nd: int = 5) -> Any:
    """Tensors and arrays to rounded lists; dicts and lists recursively; NaN to None."""
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    if isinstance(x, np.ndarray):
        return _plain(x.tolist(), nd)
    if isinstance(x, dict):
        return {str(k): _plain(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v, nd) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return round(x, nd) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _body(x: Any, b: int) -> Any:
    """Body b's element of every per-body list inside x (the plan record's lists are one value
    per body); a list of another length, or a scalar, is kept as it is."""
    if isinstance(x, dict):
        return {k: _body(v, b) for k, v in x.items()}
    if (isinstance(x, list) and x
            and all(v is None or isinstance(v, (bool, int, float)) for v in x)):
        return x[b] if b < len(x) else None
    return x                          # a list of names (planner_blind) is not per body


def causal_chain_block(model, b: int) -> Any:
    """The newest planning event of the causal chain for body b (LemLite.newest_plan_record),
    with what the trainer's recorder stamped on the plan at the latch (reached_body, order_seq,
    plan_share_of_executed on last_intention). The resolved result comes from the run's
    causal_chain.jsonl at build time, not from here."""
    if not getattr(model, "causal_chain", False):
        return "absent: --causal-chain off"
    rec = getattr(model, "newest_plan_record", None) or getattr(model, "last_plan_record", None)
    if not rec:
        return "absent: no planning event recorded since the last reset"
    if "error" in rec:
        return {"step": rec.get("step"), "error": str(rec["error"])}
    out = {k: _plain(_body(rec.get(k), b)) for k in
           ("goal", "plan0", "pred_change", "pred_wellness", "goal_distance", "wellness_rank")
           if rec.get(k) is not None}
    out.update(step=rec.get("step"), body=b, horizon=rec.get("horizon"), route=rec.get("route"),
               alternatives=list(rec.get("alternatives") or []),
               chosen_rank=_body(rec.get("chosen_rank"), b),
               # the five terms' predicted values for the chosen plan, MEAN OVER BODIES
               pred_wellness_terms_chosen=_plain(rec.get("pred_wellness_terms_chosen")))
    try:
        from lem.model.wellness import APPROXIMATE_TERMS
        out["terms"] = list(APPROXIMATE_TERMS)
    except Exception:
        pass
    li = getattr(model, "last_intention", None)
    if isinstance(li, dict) and li.get("step") == rec.get("step"):
        out.update(reached_body=li.get("reached_body"), order_seq=li.get("order_seq"),
                   plan_share_of_executed=_plain(li.get("plan_share_of_executed")))
        # the field contract of 2026-09-23 (review b1, minor 2): how far, in cm, the order the body
        # got sat from the order the planner imagined - the measurement that replaces the smoother's
        # weight; absent until the recorder writes it
        if li.get("plan_exec_distance_cm") is not None:
            out["plan_exec_distance_cm"] = _plain(li.get("plan_exec_distance_cm"))
    return out


def efference_block(model, b: int) -> Any:
    """The efference heads' newest graded tenure for body b (expected against achieved, per
    handle, cm and degrees) and the window's running means so far (read, never reset)."""
    if getattr(model, "efference", None) is None:
        return "absent: --efference-heads off"
    out: Dict[str, Any] = {"handle_names": list(getattr(model, "handle_names", []) or [])}
    t = getattr(model, "last_efference_tenure", None)
    if isinstance(t, dict) and torch.is_tensor(t.get("expected_cm")) and t["expected_cm"].shape[0] > b:
        out["tenure"] = {"step": t.get("step"), "order_began": t.get("order_began"),
                         **{k: _plain(t[k][b], 3) for k in
                            ("expected_cm", "expected_deg", "achieved_cm", "achieved_deg")
                            if torch.is_tensor(t.get(k))}}
    else:
        out["tenure"] = "absent: no command of the declared length has ended since the last reset"
    win = getattr(model, "_eff_window", None) or {}
    out["window_so_far"] = {k: round(s[0] / s[1], 4) for k, s in win.items()
                            if isinstance(s, list) and len(s) == 2 and s[1]}
    stamp = (getattr(model, "receipt", None) or {}).get("efference_heads") or {}
    out["scales"] = stamp.get("scales")
    out["weight"] = stamp.get("weight")
    out["version"] = stamp.get("version")
    return out


# the lived unit of each vestibular field: (multiplier from the stored value, unit)
def _vestibular_units() -> Dict[str, Tuple[float, str]]:
    from mimo_bridge import vestibular_adapter as VA
    deg = 180.0 / np.pi
    u: Dict[str, Tuple[float, str]] = {}
    for f in VA.VESTIBULAR_FIELDS:
        if f.startswith("head_acc"):
            u[f] = (1.0, "g")
        elif f.startswith("head_gyro"):
            u[f] = (VA.GYRO_SCALE_RADS * deg, "degrees/s")
        elif f.endswith("_vel"):
            u[f] = (VA.NECK_VEL_SCALE_RADS * deg, "degrees/s")
        elif f.endswith("_effort"):
            u[f] = (1.0, "of full strength (-1 to +1)")
        elif f.startswith("neck_"):
            u[f] = (1.0, "of the joint's range (-1 to +1)")
        elif f.startswith("trunk_up"):
            u[f] = (1.0, "unit vector, chest frame")
        elif f == "support_total":
            u[f] = (1.0, "body weights")
        elif f.startswith("support_"):
            u[f] = (1.0, "share of outside support (0 to 1)")
        elif f == "effort_power":
            u[f] = (VA.EFFORT_POWER_SCALE_W, "W")
        elif f == "effort_holding":
            u[f] = (VA.EFFORT_HOLDING_SCALE_NM, "N m")
        elif f == "limit_load":
            u[f] = (1.0, "share of joints within 2 degrees of a limit")
        elif f.startswith("capacity"):
            u[f] = (1.0, "(1 = fresh, 0 = spent)")
        else:
            u[f] = (1.0, "")
    return u


def vestibular_block(model, f) -> Any:
    """The captured body's vestibular / internal-state vector this moment, field by field in its
    lived unit, and whether the model reads it (--vestibular-sense on puts it in slot 5)."""
    v = getattr(f, "vestibular", None)
    if v is None:
        return "absent: this frame carries no vestibular vector (a transport without it)"
    try:
        from mimo_bridge import vestibular_adapter as VA
        vec = np.asarray(v, dtype=np.float64).reshape(-1)
        if vec.shape[0] == VA.LEGACY_VESTIBULAR_DIM:
            vec = np.asarray(VA.pad_legacy_vestibular(vec), dtype=np.float64)
        units = _vestibular_units()
        fields = [{"name": n, "stored": round(float(vec[i]), 5),
                   "lived": round(float(vec[i]) * units[n][0], 4), "unit": units[n][1]}
                  for i, n in enumerate(VA.VESTIBULAR_FIELDS) if i < vec.shape[0]]
    except Exception as exc:                  # one missing panel, never a lost snapshot
        return f"absent: {exc}"
    mods = list(getattr(model, "active", []) or [])
    return {"fields": fields, "read_by_model": "vestibular" in mods,
            "valid": bool(fields and fields[-1]["name"] == "valid" and fields[-1]["stored"] > 0.5)}


def wellness_channels_block(model, b: int) -> Any:
    """This moment's wellness channels for body b, the centre and scale the normalised ranking
    measures each against, and the signed contribution each makes (weight x distance from the
    centre in its own spreads) - from the goal ring's newest channel record and the running
    normaliser, read only. Controllability's gate and learning progress per sense beside them."""
    wc = getattr(model, "wellness_channels", None)
    mode = getattr(model, "wellness_rank_mode", None)
    if wc is None and mode in (None, "raw") and not getattr(model, "consequence_world_only", False):
        return "absent: no wellness channel flag is on (--wellness-rank raw, controllability and learning progress off)"
    from lem.model import wellness_channels as WC
    out: Dict[str, Any] = {"mode": mode, "curiosity_mode": getattr(model, "curiosity_mode", None),
                           "consequence_world_only": bool(getattr(model, "consequence_world_only", False)),
                           "gate_line": WC.INTEGRITY_GATE_LINE}
    # THE FIELD CONTRACT (2026-09-23): the model lane makes the channel list public
    # (rank_channel_names), adds an unranked harm row beside pain (WC.UNRANKED_ROWS), and stores
    # each ranked channel in the ring as its distance from the body's running average, the value
    # itself in wellness_channels.last_raw. Read those when present, the older names otherwise,
    # and record which.
    names = list(getattr(model, "rank_channel_names", None)
                 or getattr(model, "_rank_channel_names", None) or [])
    unranked = tuple(getattr(WC, "UNRANKED_ROWS", (WC.PAIN_ROW,)))
    ring = getattr(model, "_goal_ring_ch", None)
    raw = getattr(wc, "last_raw", None) if wc is not None else None
    detrended = getattr(model, "_rank_detrend", None) is not None
    out["stored_as"] = ("distance from this body's running average of the channel (the value "
                        "itself is wellness_channels.last_raw)" if detrended else "the channel's value")
    weights = dict(getattr(model, "rank_weights", {}) or {})
    try:
        w_now = model._rank_weights_now() if hasattr(model, "_rank_weights_now") else weights
    except Exception:
        w_now = weights
    if names and ring is not None and len(ring):
        ch = ring[-1].detach().float().cpu()                      # [C, B]
        rv = (raw.detach().float().cpu() if torch.is_tensor(raw) and raw.shape == ch.shape else None)
        if ch.shape[-1] > b:
            rows = []
            for i, n in enumerate(names):
                v = float(ch[i, b])
                if n in unranked:
                    u = float(rv[i, b]) if rv is not None else v
                    out[n] = round(u, 4) if np.isfinite(u) else None
                    continue
                c, s = (wc.normalizer.centre_scale(n) if wc is not None and wc.normalizer
                        else (float("nan"), float("nan")))
                fl = WC.FLOORS.get(n)
                den = max(s, float(fl)) if (fl is not None and np.isfinite(s)) else s
                w = float(w_now.get(n, 0.0))
                contrib = (w * (v - c) / den if all(np.isfinite(x) for x in (v, c, den)) and den > 0
                           else (0.0 if np.isfinite(c) else None))
                if rv is not None and detrended:
                    rows_raw = float(rv[i, b])
                else:
                    rows_raw = None
                rows.append({"channel": n, "value": _plain(v, 5), "raw_value": _plain(rows_raw, 5),
                             "centre": _plain(c, 5),
                             "scale": _plain(s, 5), "floor": fl, "weight": weights.get(n, 0.0),
                             "weight_in_force": w, "contribution": _plain(contrib, 4),
                             "planner_sees": WC.JUDGE_B_VIEW.get(n)})
            out["channels"] = rows
    else:
        out["channels"] = ("absent: the goal ring holds no channel record yet"
                           if names else "absent: --wellness-rank raw keeps no channel record")
    B = None
    if ring is not None and len(ring):
        B = int(ring[-1].shape[-1])
    elif getattr(model, "last_tokens", None):
        B = int(next(iter(model.last_tokens.values())).shape[0])
    if wc is not None and wc.control is not None and B:
        g = wc.control.gate()
        val = wc.control.value(B, "cpu")
        out["controllability"] = {"value": _plain(val[b] if b < B else None), "gate_open": bool(g[0]),
                                  "gate_mean": _plain(g[1]), "gate_line": _plain(g[2]), "gate_why": g[3],
                                  "last": _plain(wc.control.last)}
    if wc is not None and wc.lp is not None and B:
        per = {}
        for k in list(wc.lp.state):
            p = wc.lp.progress(k)
            rf, rs = wc.lp.ratios(k)
            if p is not None and p.shape[0] > b:
                per[k] = {"progress": _plain(p[b]), "ratio_fast": _plain(rf[b]),
                          "ratio_slow": _plain(rs[b])}
        ch_lp = wc.lp.channel(B)
        out["learning_progress"] = {"channel": _plain(ch_lp[b] if ch_lp.shape[0] > b else None),
                                    "per_sense": per}
    return out


def contact_control_block(model, b: int) -> Any:
    """This moment's foreseen touches for body b (LemLite.last_contact_control) and the
    channel's evidence gate (read only)."""
    if getattr(model, "contact_head", None) is None:
        return "absent: --contact-head off"
    lc = getattr(model, "last_contact_control", None)
    if not isinstance(lc, dict):
        return "absent: no model step has been scored since the last reset"
    bodies = lc.get("bodies") or []
    d = bodies[b] if b < len(bodies) else {}
    out = {"step": lc.get("step"), "valid": (lc.get("valid") or [None])[b] if b < len(lc.get("valid") or []) else None,
           "control_moment": bool(lc.get("control")), "gate_open": bool(lc.get("gate_open")),
           "self_predicted": d.get("self_predicted") or [], "world_predicted": d.get("world_predicted") or [],
           "would_pay": d.get("would_pay"), "pay": d.get("pay"),
           "channel_weight": float(getattr(model, "contact_control_weight", 0.0) or 0.0)}
    tr = getattr(model, "contact_tracker", None)
    if tr is not None:
        try:
            g = tr.gate()
            out.update(gate_mean=_plain(g[1]), gate_line=_plain(g[2]), gate_why=g[3])
        except Exception:
            pass
        if hasattr(tr, "gates"):
            out["gates"] = {kind: dict(open=bool(tr.gate(kind)[0]), mean=_plain(tr.gate(kind)[1]),
                                       line=_plain(tr.gate(kind)[2]), why=tr.gate(kind)[3])
                            for kind in tr.gates}
        out["margin"] = _plain(getattr(tr, "margin", None))
    return out


def synergy_block(model) -> Any:
    syn = getattr(model, "synergy", None)
    if syn is None:
        return "absent: --motor-synergy off"
    return {"dim": int(getattr(syn, "k", 0) or 0),
            "babble_sigma": _plain(getattr(model, "_synergy_sigma", None)),
            "stamp": _plain((getattr(model, "receipt", None) or {}).get("motor_synergy"))}


def _per_body(x: Any, b: int) -> Optional[int]:
    """Body b's entry of a per-body list, or the number itself when the model wrote one number."""
    if isinstance(x, (list, tuple)):
        x = x[b] if b < len(x) else None
    if torch.is_tensor(x):
        x = x.reshape(-1)[b].item() if x.numel() > b else None
    try:
        return None if x is None or isinstance(x, bool) else int(x)
    except (TypeError, ValueError):
        return None


def _goal_index_now(li: Any, b: int, step_now: int, ring_len: int) -> Dict[str, Any]:
    """Where the planner's goal sits in the goal ring AS IT STANDS at the snapshot.

    The model records its pick in last_intention: goal_age_steps (0 = the newest moment of the
    ring at planning time) and goal_ring_index, one per body or one number, beside the plan's own
    step. The ring gains one moment every step (LemLite.step appends while the planner exists), so
    a goal that was a moments old at step p is a + (step_now - p) moments old now. Returns
    {"index": ring index now or None, "fields": what the snapshot records about it}; a snapshot
    whose model records no pick says so ("goal_index_absent") and names no goal."""
    if not isinstance(li, dict):
        return {"index": None, "fields": {"goal_source": "not recorded",
                                          "goal_index_absent": "no plan has been made since the last reset"}}
    age = _per_body(li.get("goal_age_steps"), b)
    idx = _per_body(li.get("goal_ring_index"), b)
    plan_step = li.get("step")
    if age is None and idx is None:
        return {"index": None, "fields": {
            "goal_source": "not recorded",
            "goal_index_absent": "this run's model does not record which remembered model step its "
                                 "planner chose (last_intention carries no goal_ring_index or "
                                 "goal_age_steps), so the page names no goal"}}
    moved = (step_now - int(plan_step)) if isinstance(plan_step, (int, float)) else None
    fields: Dict[str, Any] = {"goal_source": "recorded", "goal_plan_step": plan_step,
                              "goal_age_at_plan": age, "goal_ring_index_at_plan": idx}
    if age is not None and moved is not None and moved >= 0:
        now = age + moved
        fields["goal_age_now"] = now
        if now < ring_len:
            fields["goal_ring_index_now"] = ring_len - 1 - now
            return {"index": ring_len - 1 - now, "fields": fields}
        fields["goal_index_absent"] = ("the goal model step has left the ring since the plan (it was "
                                       f"{age} model steps old then and {now} now, in a ring of "
                                       f"{ring_len})")
        return {"index": None, "fields": fields}
    if idx is not None and moved == 0 and 0 <= idx < ring_len:
        fields["goal_ring_index_now"] = idx
        return {"index": idx, "fields": fields}
    fields["goal_index_absent"] = ("the plan was made at another step and the record carries no "
                                   "age to follow the goal through the ring since, so the page "
                                   "names no goal")
    return {"index": None, "fields": fields}


def reader_ceiling(model) -> float:
    """The most the entity reader can add to a stored value: its applied weight x the sum of the
    positive proven links (presence is at most 1). 0 while its gate is shut."""
    sv = getattr(model, "_self_value", None)
    if sv is None:
        return 0.0
    try:
        return float(sv.applied_weight) * float(sv._weights().clamp(min=0.0).sum())
    except Exception:
        return 0.0


# ----------------------------------------------------------------- the snapshot
def build_snapshot(model, frames: Sequence, res, *,
                   run: str, step: int, phase: Optional[str], dev: Dict[str, Any],
                   weights: str, body: int = 0,
                   eye_layout: Optional[Sequence[Tuple[str, int]]] = None,
                   prev_executed: Optional[np.ndarray] = None,
                   held_order: Optional[np.ndarray] = None,
                   order_seq: Optional[int] = None, order_age: Optional[int] = None,
                   age_months: Optional[float] = None,
                   flags: Optional[Dict[str, Any]] = None,
                   replay_info: Optional[Dict[str, Any]] = None,
                   order_sent: Optional[np.ndarray] = None,
                   rest_window: Optional[bool] = None,
                   check_invariants: bool = False) -> Dict[str, Any]:
    """Assemble one snapshot dict (schema.SCHEMA_VERSION) for `body`. Raises only on a model
    that has never stepped; every optional absence becomes a labeled reason string.

    order_sent / rest_window (train_mimo.explain_rest_fields hands them over when this
    signature carries them): the order the body was really sent this interval and whether a
    scheduled rest window replaced its position and rotation with the achieved pose - so the
    held (proposed) order and the sent one are never read as one."""
    t0 = time.perf_counter()
    b = int(body)
    f = frames[b] if b < len(frames) else frames[0]

    # attention (the one GPU-touching part)
    raw_layers, cap_last = capture_attention_maps(model, body=b,
                                                  check_invariants=check_invariants)
    layers_out = []
    for lw in raw_layers:
        entry: Dict[str, Any] = {
            "stage1": None if lw["stage1"] is None else quantize(lw["stage1"]),
            "stage2": None if lw["stage2"] is None else quantize(lw["stage2"]),
        }
        if lw["bridge"] is not None:
            agg = _bridge_aggregates(lw["bridge"], model.ctx_len, model.n_mod)
            entry["stage3_rows"] = quantize(agg["rows"])
            entry["stage3_pair"] = _lst(agg["pair"])
            entry["stage3_lag"] = _r(agg["lag"])
            entry["stage3_lag_forward"] = _r(agg["lag_forward"])
            entry["stage3_direction_share"] = _r(agg["direction_share"])
        else:
            entry["stage3_rows"] = entry["stage3_pair"] = entry["stage3_lag"] = None
            entry["stage3_lag_forward"] = entry["stage3_direction_share"] = None
        layers_out.append(entry)

    # tokens
    raw_tok = {m: _lst(v[b]) for m, v in (model.last_tokens or {}).items()}
    fed_differs = False
    if model.last_tokens_fed is not None and model.last_tokens is not None:
        for m in model.last_tokens:
            fed = model.last_tokens_fed.get(m)
            if fed is not None and not torch.allclose(
                    fed[b].reshape(-1), model.last_tokens[m][b].reshape(-1), atol=1e-6):
                fed_differs = True
                break
    ctx_t = model.last_context[b] if model.last_context is not None else None
    ctx_rms = None
    if ctx_t is not None:
        r = ctx_t.detach().float().reshape(-1, model.n_mod, model.d_model)
        ctx_rms = _lst(r.pow(2).mean(-1).sqrt())          # [ctx-1, n_mod]

    # predictions
    nf = getattr(model, "_nf_pred", None)
    of_now = ({m: _lst(v[b]) for m, v in nf.items() if v.shape[0] > b}
              if nf else "absent: first step after reset, or --next-frame-loss off")
    of_next = ({m: _lst(v[b]) for m, v in (model.last_predictions or {}).items()
                if v.shape[0] > b}
               if model.last_predictions else "absent: model has not stepped")

    # senses
    senses: Dict[str, Any] = {
        "contacts": int(getattr(f, "n_contacts", 0)),
        "task": _r(dict((f.meta.get("task") or {}))),
        "sim_time": float(getattr(f, "sim_time", 0.0)),
    }
    try:
        from mimo_bridge.lem_model import touch_point_cloud
        imp = f.tokenized.touch_impulses
        senses["touch_cloud"] = _lst(touch_point_cloud(
            imp, getattr(f, "touch_pos", None), getattr(f, "touch_self", None),
            getattr(f, "touch_contact", None)))
        senses["touch_cloud_doc"] = TOUCH_CLOUD_DOC
    except Exception as e:
        senses["touch_cloud"] = f"absent: {e}"
    try:
        senses["bones"] = _lst(np.asarray(f.bones))
    except Exception as e:
        senses["bones"] = f"absent: {e}"
    if eye_layout and getattr(f, "vision_patches", None) is not None:
        try:
            vp = f.vision_patches
            if torch.is_tensor(vp):
                vp = vp.detach().float().cpu().numpy()
            senses["eyes"] = reassemble_eyes(np.asarray(vp), eye_layout)
        except ValueError as e:
            senses["eyes"] = f"absent: {e}"
    else:
        senses["eyes"] = ("absent: vision stale this step (previous token re-used)"
                          if eye_layout else "absent: no eye layout given")

    # motor
    def _row(a, label):
        if a is None:
            return f"absent: {label}"
        arr = np.asarray(a)
        return _lst(arr[b] if arr.ndim == 3 else arr)
    motor = {
        # the model's own list, not the 24-name base constant: trunk-body layouts
        # order 26 handles, and the base list silently dropped the last two rows
        "handle_names": list(getattr(model, "handle_names", None) or HANDLE_NAMES),
        "motor_out": _lst(res.motor_out[b]),
        "motor_for_action": _lst(res.motor_for_action[b]),
        "prev_executed": _row(prev_executed, "no executed command yet (first interval)"),
        "held_order": _row(held_order, "no held command yet"),
        "order_seq": order_seq, "order_age": order_age,
        "gaze_az_el": (None if res.gaze_out is None else _lst(res.gaze_out[b])),
        "confidence": _r(float(res.motor_confidence)),
        "wellness_grad_norm": _r(float(res.wellness_grad_norm)),
        # the order really sent this interval (a scheduled rest window replaces its position
        # and rotation with the achieved pose) - schema v3
        "order_sent": _row(order_sent, "not handed to the capture by this producer"),
        "rest_window": None if rest_window is None else bool(rest_window),
    }
    # INTEGRITY: one name per order row. A 24-name list was once stamped beside 26-row
    # orders and the viewer repaired it downstream, so no other reader of the file could
    # see the mismatch. train_mimo.py wraps this call and logs the failure, so a mismatch
    # costs this one snapshot and never the training step.
    if len(motor["handle_names"]) != len(motor["motor_out"]):
        raise ValueError(
            f"snapshot motor block: {len(motor['handle_names'])} handle names for "
            f"{len(motor['motor_out'])} command rows - the name list is not this body's")

    # drives
    wg = model.last_wellness_grounded or {}
    drives = {
        "wellness_body": {k: _r(float(v[b].float().mean()))
                          for k, v in wg.items() if v.shape[0] > b},
        # WHICH PAIRING PRODUCED prediction_error THIS STEP (P076). lem_model sets
        # last_pe_source every step. The authoritative pairing is the delayed causal
        # prediction against raw tokens; on the first step of a run, of a resume, and
        # after every reset_transition_state / reset_context the delayed prediction
        # does not exist and the same-frame pair is used, which with
        # residual_prediction on reads 1 - clamp(mean(head_output^2)) exactly
        # (measured: difference 0.0 over 4 bodies) - whether the heads predicted NO
        # CHANGE, not whether they predicted correctly. Published so the page never
        # draws the two as one bar.
        "prediction_error_source": getattr(model, "last_pe_source", None),
        # WHICH READING OF THE BODY MAP the coherence term carried this step. --wellness-
        # coherence picks between the map's EXTENT (the share of everything it watches that
        # it calls its own) and its CERTAINTY (how sure it is either way), and with no body
        # map running the term is a third quantity again - the cosine likeness of prediction
        # to arrival. The flag's own help has said "recorded in the snapshot as
        # last_coherence_source" since it was added; the snapshot did not carry it.
        "coherence_source": getattr(model, "last_coherence_source", None),
        "wellness_batch_mean": _r(dict(res.wellness)),
        "curiosity_drive": _r(float(res.curiosity_drive)),
        "curiosity_zone": (model.curiosity.zone(res.mean_pe)
                           if getattr(model, "curiosity", None) is not None else None),
        "movement": _r({k: float(v) for k, v in (model.movement.last or {}).items()
                        if isinstance(v, (int, float))})
                    if getattr(model, "movement", None) is not None and model.movement.last
                    else None,
        "agency": _r(dict(model.last_agency or {})) or None,
        "td_gate": _r(dict(model.last_gate or {})) or None,
        "tenure": {"pe": _r(dict(model.last_tenure_pe or {})),
                   "gap": _r(dict(model.last_tenure_gap or {})),
                   "tok": _r(dict(model.last_tenure_tok or {}))},
    }

    # newer components (schema v2): scalar summaries, each absent with a reason
    def _scalars(d):
        out: Dict[str, Any] = {}
        for k, v in (d or {}).items():
            if isinstance(v, (bool, int, float, str)) or v is None:
                out[k] = v
            elif isinstance(v, dict):
                inner = {ik: iv for ik, iv in v.items()
                         if isinstance(iv, (bool, int, float, str)) or iv is None}
                if inner:
                    out[k] = inner
            elif (isinstance(v, (list, tuple))
                  and all(isinstance(x, (int, float)) for x in v)):
                out[k] = list(v)
        return _r(out)

    memory_block = (_scalars(model.last_memory)
                    if getattr(model, "persistent_memory", None) is not None
                    and model.last_memory
                    else "absent: persistent memory off or not yet observed")
    # THE TOKEN THE RECALLED BIAS LANDS ON (2026-09-19). The memory diagram drew "a third
    # to a half the size of that token" on every run's page; that was H81's measurement
    # (0.19 to 0.28 against 0.59). bias_rms alone cannot be placed - H84's snapshots read
    # 3.0 to 4.0 and nothing in them says how big the token under it is. The re-forward
    # above already holds the core's newest-moment output WITH the bias added (cap_last,
    # held equal to the live pass by test_reforward_matches_live_step) and the bias is
    # model._memory_bias_current, so the token's own size costs one subtraction. Both are
    # THIS body's; bias_rms above is the trainer's figure over every body. Under
    # --self-state perceive the same tensor also carries the self-perception bias.
    if isinstance(memory_block, dict):
        try:
            mb = getattr(model, "_memory_bias_current", None)
            if mb is not None and mb.shape[0] > b and tuple(mb[b].shape) == tuple(cap_last.shape):
                bias_b = mb[b].detach().float()
                landing = cap_last.detach().float() - bias_b.to(cap_last.device)
                memory_block["bias_rms_body"] = _r(float(bias_b.pow(2).mean().sqrt()))
                memory_block["landing_token_rms"] = _r(float(landing.pow(2).mean().sqrt()))
        except Exception as exc:          # one missing row, never a lost snapshot
            memory_block["landing_token_rms_absent"] = f"absent: {exc}"
    self_block = (_scalars(model.last_self_state)
                  if getattr(model, "self_state", None) is not None
                  and model.last_self_state
                  else "absent: self perception off or not yet observed")
    hs = getattr(model, "_hself", None)
    hself_blk = (hself_block(hs.stats(), _scalars) if hs is not None
                 else "absent: hierarchical self off")
    # incorporation_stats() returns None when --body-schema-updates is off or no body map
    # has been built yet; it is a labelled absence here, the way every other optional block
    # is, so the panel can say which of the two it is rather than drawing nothing.
    inc = (model.incorporation_stats()
           if hasattr(model, "incorporation_stats") else None)
    incorporation_blk = (_r(inc) if isinstance(inc, dict) and inc
                         else "absent: the body map is not being updated "
                              "(--body-schema-updates off), or it has not been built yet")
    lv = getattr(model, "last_voice", None)
    voice_block = ({"rms": _r(float(lv[b].float().pow(2).mean().sqrt())),
                    "min": _r(float(lv[b].min())), "max": _r(float(lv[b].max())),
                    "n_controls": int(lv.shape[-1]),
                    # the number format these are carried in: under the run's bf16
                    # autocast, neighbouring values near 0.5 are 0.004 apart, so the
                    # page can say how many of the printed decimals mean anything
                    "dtype": str(lv.dtype).replace("torch.", ""),
                    "world_effect": "none: no vocal tract is wired into the crib "
                                    "world yet, so this output makes no sound"}
                   if lv is not None and lv.shape[0] > b
                   else "absent: voice head off")
    cg = ((getattr(f, "meta", None) or {}).get("caregiver")
          or (getattr(f, "info", None) or {}).get("caregiver"))
    caregiver_block = _scalars(cg) if cg else "absent: caregiver off"

    ring = getattr(model, "goal_ring", None)
    intention = {
        "last": _r(dict(model.last_intention)) if model.last_intention else None,
        "goal_ring_size": len(ring) if ring is not None else 0,
        "best_remembered_wellness": (_r(float(max(float(w[b]) for _, w in ring)))
                                     if ring else None),
        "scale_report": _r(dict(model.last_scale_report))
                        if model.last_scale_report else None,
    }
    # P119: the winner's felt value and its curiosity factor, side by side. The stored
    # value is (w0 + movement + agency + consequence) x (1 + drive), so w0 alone cannot
    # separate the factor from the additive drives; the per-moment drive the model keeps
    # beside the ring makes the split exact.
    # WHICH MOMENT IS THE GOAL (review b3-wellness, minor, 2026-09-22). This used to take the ring
    # moment with the largest RAW stored value NOW as the goal. Under --wellness-rank normalized
    # the planner picks the largest NORMALISED value instead (in the review's measurement the
    # oldest moment, generally not the raw best), and at any ranking the ring has moved on since
    # the plan (on segment 31's newest snapshot the plan was 9 moments older than the capture).
    # The model now records its pick (last_intention goal_ring_index / goal_age_steps, one per
    # body or one number); the goal is read from that, mapped to the ring as it stands now, and a
    # snapshot without the record says the goal is not recorded - never a guess from the raw best.
    drv = getattr(model, "_goal_ring_drive", None)
    if ring and drv is not None and len(drv) == len(ring):
        import math as _math
        stored = [float(w[b]) for _, w in ring]
        felt_all = [stored[t] / (1.0 + float(drv[t])) for t in range(len(stored))]
        j = max(range(len(felt_all)), key=felt_all.__getitem__)
        gi = _goal_index_now(model.last_intention, b, int(step), len(ring))
        intention.update(gi["fields"])
        i = gi["index"]
        mv = getattr(model, "movement", None)
        # 5.0 is the five lem/model/wellness.py formulas at their tops. The sixth grounded
        # term, self_touch, reaches its own launch weight and no more (lem_model.py
        # _self_touch_wellness returns rise x weight x gate, rise <= 1), so the ceiling
        # gained that weight the day the term did; at --self-touch-wellness 0 it adds 0.0.
        # The entity READER adds weight x the sum of (proven link x presence) once its gate
        # is open (lem_model.py, the self._self_value.value line before the curiosity factor),
        # so the ceiling carries the most that can add; and under --curiosity-mode additive the
        # drive is ADDED at its channel weight (drive <= 1) instead of multiplying.
        parts = {"five_formulas": 5.0,
                 "self_touch": float(getattr(model, "self_touch_wellness_weight", 0.0)),
                 "movement": ((0.5 + float(getattr(mv, "bonus", 0.0)) * _math.tanh(3.0))
                              if mv is not None else 0.0),
                 "agency": ((float(model.agency_weight) + float(model.agency_prior_w0))
                            if getattr(model, "agency_drive", False) else 0.0),
                 "consequence": 3.0 * float(getattr(model, "consequence_weight", 0.0)),
                 "reader": reader_ceiling(model),
                 "curiosity_added": (float((getattr(model, "rank_weights", {}) or {}).get("curiosity", 0.0))
                                     if (getattr(model, "curiosity_mode", "multiply") == "additive"
                                         and getattr(model, "curiosity_goal", False)) else 0.0)}
        ceiling = sum(parts.values())
        intention.update({"best_felt_value": _r(felt_all[j]),
                          "stored_value_ceiling_no_curiosity": _r(ceiling),
                          "stored_value_ceiling_parts": _r(parts)})
        if i is not None:
            factor = 1.0 + float(drv[i])
            intention.update({"goal_stored_value": _r(stored[i]),
                              "goal_felt_value": _r(stored[i] / factor),
                              "goal_curiosity_factor": _r(factor),
                              "goal_wins_without_curiosity": bool(i == j)})
    elif ring:
        intention["goal_curiosity_factor_absent"] = "the model keeps no per-model-step drive beside the ring (old code)"

    snap: Dict[str, Any] = {
        "header": {
            "schema_version": SCHEMA_VERSION, "run": run, "step": int(step),
            "timestamp": time.time(), "phase": phase, "age_months": age_months,
            "dev_limits": _r({k: v for k, v in (dev or {}).items()
                              if isinstance(v, (int, float, str))}),
            "source_mask": dict(res.source_mask), "n_envs": len(frames),
            "body": b, "weights": weights, "capture_ms": None,
            "flags": _r(flags or {}),
        },
        "model": model.receipt,
        "tokens": {
            "raw": raw_tok,
            "fed": ({m: _lst(v[b].reshape(-1)) for m, v in model.last_tokens_fed.items()}
                    if fed_differs else "same as raw (top-down gate closed or off)"),
            "context_rms": ctx_rms,
            "token_temporal_std": _r(model.token_temporal_std()),
        },
        "predictions": {
            "of_now": of_now,
            "actual_now_is": "tokens.raw (the delayed lesson's own pairing, review P4)",
            "error_of_now": _r(dict(model.last_next_pe or {})),
            "of_next": of_next,
            "recon_dec_error": _r(dict(model.last_recon_dec or {})),
            "recon_pe": _r(float(res.recon_pe)) if res.recon_pe == res.recon_pe else None,
            "causal_pe": _r(float(res.causal_pe)) if res.causal_pe == res.causal_pe else None,
        },
        "attention": {
            "modalities": list(model.active), "ctx": model.ctx_len,
            "seq": model.ctx_len * model.n_mod, "heads": "averaged",
            "inputs": "tokens.fed vs last_context (the live forward's own inputs)",
            "layers": layers_out,
        },
        "senses": senses,
        "motor": motor,
        "drives": drives,
        "intention": intention,
        "memory": memory_block,
        "self": self_block,
        # UEMA 11.2 object incorporation: how much of the body, and of each outside thing,
        # the body map calls its own. The trainer has written this into every stats record
        # since the map grew object columns; the page had no panel for it, so the toy's
        # number, its control and whether the world lets the body move the toy at all were
        # visible only to someone reading the log by hand.
        "incorporation": incorporation_blk,
        "hself": hself_blk,
        # the same two levels over NAMED things, and what the reader is doing with them
        "entity_self": (model.entity_self_stats()
                        if hasattr(model, "entity_self_stats") else None),
        "voice": voice_block,
        "caregiver": caregiver_block,
        # schema v3 (2026-09-22): the gain-of-function build's features, each a labelled
        # absence when its flag is off
        "causal_chain": causal_chain_block(model, b),
        "efference": efference_block(model, b),
        "vestibular": vestibular_block(model, f),
        "wellness_channels": wellness_channels_block(model, b),
        "contact_control": contact_control_block(model, b),
        "synergy": synergy_block(model),
        "losses": {
            "loss": _r(float(res.loss)),
            "breakdown": _r(dict(res.breakdown)),
            "grad_norm": _r(float(res.grad_norm)),
            "imagination_loss": (None if res.imagination_loss is None
                                 else _r(float(res.imagination_loss))),
            "scope": "this live step only - replay re-study trains separately",
            "replay": _r(replay_info) if replay_info else None,
            "timings_ms": _r(dict(res.timings_ms)),
        },
    }
    snap["header"]["capture_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    return snap
