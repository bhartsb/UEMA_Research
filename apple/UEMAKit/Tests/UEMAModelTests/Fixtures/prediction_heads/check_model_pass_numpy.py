# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A float64 numpy copy of the Swift model port of work item WI-35 (Sources/UEMAModel/TransformerCore.swift,
Encoders/*.swift, PredictionHeads.swift, ForwardModel.swift), run on the WI-34 fixtures with the released weights,
to check the layer sequence before the Swift code is compiled on a Mac.

INPUT: weights/b1_continue_B1K_skip13_492800/run/checkpoints/ckpt_step_494025/model.safetensors (read without torch:
the 8-byte header length, the JSON header, the raw data), Fixtures/model_pass/<state>.safetensors (WI-34) and
Sources/UEMAModel/Resources/prediction_head_centres.json. OPERATION: every function follows the Swift function of the
same role (Linear, LayerNorm eps 1e-5, exact GELU, PyTorch-layout attention, the Swin blocks with zero padding and the
half-window roll, the cross-patch encoders with the liveness repair's all-element pool path, the audio convolutions,
the multi-layer perceptron encoders, the three-stage transformer with the memory bias, the prediction heads with the
revived units and the all-slot path). OUTPUT: per state, the largest absolute difference against the fixture of each
encoder's features, pooled vector and token, the latents, and each predicted token, with and without the all-slot path.
Measured November 4, 2026: every token within 1.2e-6 and every predicted token within 1.4e-6 on the three states.

Run from D:/UEMA_Research (numpy and scipy; no torch):
    python -I apple/UEMAKit/Tests/UEMAModelTests/Fixtures/prediction_heads/check_model_pass_numpy.py step_000492800
"""
import json, os, struct, sys
import numpy as np
from scipy.special import erf

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *([".."] * 6)))
W_PATH = os.path.join(REPO, "weights", "b1_continue_B1K_skip13_492800", "run", "checkpoints", "ckpt_step_494025", "model.safetensors")
FIX = os.path.join(REPO, "apple", "UEMAKit", "Tests", "UEMAModelTests", "Fixtures", "model_pass", "%s.safetensors")
DT = {"F32": np.float32, "F64": np.float64, "BOOL": np.bool_, "I64": np.int64}


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        out = {}
        data = open(path, "rb").read()
        for k, v in h.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            out[k] = np.frombuffer(data[base + a: base + b], dtype=DT[v["dtype"]]).reshape(v["shape"]).astype(np.float64 if v["dtype"] != "BOOL" else np.bool_)
        return out, h.get("__metadata__", {})


W, _ = load(W_PATH)
CENTRE = {k: np.asarray(v["centre_mean"], dtype=np.float64) for k, v in json.load(open(
    os.path.join(REPO, "apple", "UEMAKit", "Sources", "UEMAModel", "Resources", "prediction_head_centres.json")))["layers"].items()}
POOLPATH = True
REVIVE = True


def T(k):
    return W[k]


def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


def linear(x, p, bias):
    y = x @ T(p + "weight").T
    return y + T(p + "bias") if bias else y


def ln(x, p=None):
    c = x - x.mean(-1, keepdims=True)
    v = (c * c).mean(-1, keepdims=True)
    y = c / np.sqrt(v + 1e-5)
    if p is not None:
        y = y * T(p + "weight") + T(p + "bias")
    return y


def softmax(s):
    s = s - s.max(-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(-1, keepdims=True)


def mha(q_in, k_in, v_in, p, heads, bias, mask=None):
    D = q_in.shape[-1]
    w = T(p + "in_proj_weight")
    wq, wk, wv = w[:D], w[D:2 * D], w[2 * D:]
    if bias:
        b = T(p + "in_proj_bias")
        bq, bk, bv = b[:D], b[D:2 * D], b[2 * D:]
    else:
        bq = bk = bv = 0
    B, L, _ = q_in.shape
    S = k_in.shape[1]
    hd = D // heads
    q = (q_in @ wq.T + bq).reshape(B, L, heads, hd).transpose(0, 2, 1, 3)
    k = (k_in @ wk.T + bk).reshape(B, S, heads, hd).transpose(0, 2, 1, 3)
    v = (v_in @ wv.T + bv).reshape(B, S, heads, hd).transpose(0, 2, 1, 3)
    s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    if mask is not None:
        s = s + mask
    o = (softmax(s) @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
    return linear(o, p + "out_proj.", bias)


def prenorm(x, p, n1, n2, heads, bias, mask=None):
    h = ln(x, p + n1 + ".")
    a = mha(h, h, h, p + "attn.", heads, bias, mask) + x
    return linear(gelu(linear(ln(a, p + n2 + "."), p + "ffn.0.", bias)), p + "ffn.3.", bias) + a


def unit(x):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + 1e-24)


def vision(patches, angles):
    p = "encoder_registry.encoders.vision."
    x = linear(patches, p + "patch_embed.", False) + linear(gelu(linear(angles, p + "angular_pos_enc.mlp.0.", True)), p + "angular_pos_enc.mlp.2.", True)
    for i in range(4):
        B, N, D = x.shape
        ws = 7
        pad = (ws - N % ws) % ws
        y = np.concatenate([x, np.zeros((B, pad, D))], 1) if pad else x
        sh = ws // 2 if i % 2 == 1 else 0
        if sh:
            y = np.roll(y, -sh, 1)
        Np = N + pad
        o = prenorm(y.reshape(B * Np // ws, ws, D), p + "blocks.%d." % i, "norm1", "norm2", 8, False).reshape(B, Np, D)
        if sh:
            o = np.roll(o, sh, 1)
        x = o[:, :N]
    feats = x
    q = unit(ln(T(p + "pool.query"), p + "pool.norm.")) * 192 ** 0.25
    k = unit(feats) * 192 ** 0.25
    return mha(q, k, feats, p + "pool.attn.", 4, False), feats


def crosspatch(rows, sense, pos):
    p = "encoder_registry.encoders.%s." % sense
    x = linear(gelu(linear(rows, p + "patch_mlp.0.", False)), p + "patch_mlp.2.", False)
    if pos:
        x = x + T(p + "patch_pos")
    for i in range(2):
        x = prenorm(x, p + "layers.%d." % i, "norm", "ffn_norm", 4, False)
    q = ln(T(p + "pool_query"), p + "pool_norm.")
    o = mha(q, x, x, p + "pool_attn.", 4, False)
    if POOLPATH:
        lp = "liveness_repair.paths.%s." % sense
        z = linear(ln(x, lp + "norm."), lp + "down.", False)
        o = o + linear(z.reshape(x.shape[0], 1, -1), lp + "up.", True)
    return o, x


def audio(spec, spatial):
    p = "encoder_registry.encoders.audio."
    x = spec  # [B, L, C]
    for idx in (0, 3, 6, 9):
        w = T(p + "spectral_conv.%d.weight" % idx)  # [O, I, K]
        xp = np.concatenate([np.zeros((x.shape[0], 1, x.shape[2])), x, np.zeros((x.shape[0], 1, x.shape[2]))], 1)
        L = x.shape[1]
        out = sum(xp[:, k:k + L] @ w[:, :, k].T for k in range(3))
        x = gelu(out)
    s = linear(x.mean(1), p + "spectral_proj.", False)
    B = spec.shape[0]
    bands = spec.reshape(B, 16, 16, 2).mean(2).reshape(B, 32)
    s = s + linear(bands, p + "band_proj.", False)
    sp = linear(spatial, p + "spatial_proj.", False)
    g = 1 / (1 + np.exp(-linear(np.concatenate([s, sp], -1), p + "gate.", False)))
    return ln(g * s + (1 - g) * sp, p + "norm.")[:, None, :]


def mlp(x, sense, bias):
    p = "encoder_registry.encoders.%s." % sense
    h = x
    for i in range(3):
        h = gelu(ln(linear(h, p + "mlp.%d." % (4 * i), bias), p + "mlp.%d." % (4 * i + 1)))
    return linear(h, p + "mlp.12.", bias)[:, None, :]


def core(tokens, context, bias):
    p = "forward_model."
    cur = tokens + T(p + "modality_embedding.weight")
    temp = np.repeat(T(p + "temporal_embedding.weight")[:, None, :], 6, 1).reshape(768, 192)
    x = np.concatenate([context, cur], 1) + temp
    mask = np.triu(np.full((128, 128), -np.inf), 1)
    B = x.shape[0]
    for i in range(4):
        b = p + "blocks.%d." % i
        y = x.reshape(B * 128, 6, 192)
        h = ln(y, b + "cm_norm.")
        x = (mha(h, h, h, b + "cm_attn.", 8, True) + y).reshape(B, 768, 192)
        y = x.reshape(B, 128, 6, 192).transpose(0, 2, 1, 3).reshape(B * 6, 128, 192)
        h = ln(y, b + "ct_norm.")
        x = (mha(h, h, h, b + "ct_attn.", 8, True, mask) + y).reshape(B, 6, 128, 192).transpose(0, 2, 1, 3).reshape(B, 768, 192)
        h = ln(x, b + "cmb_norm.")
        x = mha(h, h, h, b + "cmb_attn.", 8, True) + x
        x = linear(gelu(linear(ln(x, b + "ffn_norm."), b + "ffn.0.", True)), b + "ffn.3.", True) + x
    x = ln(x, p + "final_norm.")
    if bias is not None:
        x = np.concatenate([x[:, :762], x[:, 762:] + bias], 1)
    return x, x[:, 762:], cur


def heads(last, cur, allslot=True):
    out = {}
    c = T("persistent_memory.readout.bias").reshape(6, 192)
    xc = last - c - T("heads_all_slot_path.centre_mean")
    for s in ("vision", "tactile", "audio", "proprioception", "vestibular"):
        p = "prediction_heads.heads.%s." % s
        q = ln(T(p + "query"), p + "norm.")
        a = mha(q, last, last, p + "attn.", 4, True).reshape(last.shape[0], 192)
        scale = float(W[p + "mlp.0.input_scale"]) if (p + "mlp.0.input_scale") in W else 1.0
        pre = linear(a / scale, p + "mlp.0.", True)
        rk = "liveness_repair.rows.prediction_heads__heads__%s__mlp__0." % s
        if REVIVE and (rk + "mask") in W:
            z = linear(ln(a - CENTRE["prediction_heads__heads__%s__mlp__0" % s], rk + "norm."), rk + "lin.", True)
            pre = np.where(W[rk + "mask"], z, pre)
        h = linear(gelu(pre), p + "mlp.2.", True)
        h = h + cur[s]
        if allslot:
            ap = "heads_all_slot_path.paths.%s." % s
            u = linear(ln(xc, ap + "norm."), ap + "down.", False).reshape(last.shape[0], 192)
            pre = linear(u, ap + "net.0.", True)
            rk = "liveness_repair.rows.heads_all_slot_path__paths__%s__net__0." % s
            if REVIVE and (rk + "mask") in W:
                z = linear(ln(u - CENTRE["heads_all_slot_path__paths__%s__net__0" % s], rk + "norm."), rk + "lin.", True)
                pre = np.where(W[rk + "mask"], z, pre)
            h = h + linear(gelu(pre), ap + "net.2.", True) / np.sqrt(192)
        out[s] = h
    return out


def run(state):
    F, meta = load(FIX % state)
    rep = {}
    def cmp(name, a, b):
        rep[name] = float(np.abs(np.asarray(a).reshape(-1) - np.asarray(b).reshape(-1)).max())
    pooled = {}
    pooled["vision"], vf = vision(F["input/vision/patches"], F["input/vision/angular_pos"])
    cmp("features/vision", vf, F["encoder_features/vision"])
    pooled["tactile"], tf = crosspatch(F["input/tactile"], "tactile", False)
    cmp("features/tactile", tf, F["encoder_features/tactile"])
    pooled["audio"] = audio(F["input/audio/spectral"], F["input/audio/spatial"])
    pooled["proprioception"] = mlp(F["input/proprioception"], "proprioception", False)
    pooled["motor"], mf = crosspatch(F["input/motor"], "motor", True)
    cmp("features/motor", mf, F["encoder_features/motor"])
    pooled["vestibular"] = mlp(F["input/vestibular"], "vestibular", True)
    toks = {}
    for s in ("vision", "tactile", "audio", "proprioception", "motor", "vestibular"):
        cmp("pooled/" + s, pooled[s], F["encoder_output/" + s])
        toks[s] = ln(pooled[s]).reshape(1, 192)
        cmp("token/" + s, toks[s], F["token/" + s])
    tokens = np.stack([toks[s] for s in ("vision", "tactile", "audio", "proprioception", "motor", "vestibular")], 1)
    lat, last, cur = core(tokens, F["forward/context_in"], F["forward/memory_bias"])
    cmp("latents", lat, F["forward/latents"])
    cmp("last_step", last, F["forward/last_step"])
    cmp("context_update", cur, F["forward/context_update"])
    pr = heads(last, toks)
    for s, v in pr.items():
        cmp("predicted/" + s, v, F["predicted/" + s])
    pr0 = heads(last, toks, allslot=False)
    for s, v in pr0.items():
        cmp("predicted_without_allslot/" + s, v, F["predicted/" + s])
    cmp("handle_state_vs_input_motor10", F["command/handle_state"], F["input/motor"][..., :10])
    rep["memory_bias_max_abs"] = float(np.abs(F["forward/memory_bias"]).max())
    rep["memory_bias_minus_readout_bias_max_abs"] = float(np.abs(F["forward/memory_bias"].reshape(-1) - T("persistent_memory.readout.bias")).max())
    return rep


if __name__ == "__main__":
    for st in sys.argv[1:]:
        print(st, json.dumps(run(st), indent=1))
