# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The liveness parts by the code's names, with the code path beside the name wherever a chart, a bar or a
table column is titled (the operator's reading gate: a code name the operator must decode gets its plain
meaning beside it; and one name per part on both pages). Until 2026-09-25 the development dashboard titled
its history charts "Units responding: recon_decoders.vision.0", the explainer labelled its liveness bars
and chart lines with bare code paths, and the Model health card printed 37 of the liveness parts
production's record names as code paths (review of 2026-09-25, finding 8). The names are the development
dashboard's (_part, _response_name), read from the model code; the explainer and the history charts use
the same ones."""
import json
import os
import re

from mimo_bridge.tools import build_explainer as BX
from mimo_bridge.tools import development_dashboard as DD
from mimo_bridge.tools import lineage_series as LS

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# the liveness parts production's newest record named on 2026-09-25 (H84 seg40, the segment resumed at
# 3:48 PM, record of step 414,200, read-only): the whole-model block's layers and pools, and each
# module's own report
PRODUCTION_LAYERS = (
    "contact_head.ctx1", "efference.body_mlp.0", "efference.body_mlp.2",
    "encoder_registry.encoders.motor.layers.0.ffn.0", "encoder_registry.encoders.motor.layers.1.ffn.0",
    "encoder_registry.encoders.motor.patch_mlp.0", "encoder_registry.encoders.tactile.layers.0.ffn.0",
    "encoder_registry.encoders.tactile.layers.1.ffn.0", "encoder_registry.encoders.tactile.patch_mlp.0",
    "encoder_registry.encoders.vision.angular_pos_enc.mlp.0", "encoder_registry.encoders.vision.blocks.0.ffn.0",
    "encoder_registry.encoders.vision.blocks.1.ffn.0", "encoder_registry.encoders.vision.blocks.2.ffn.0",
    "encoder_registry.encoders.vision.blocks.3.ffn.0", "forward_model.blocks.0.ffn.0",
    "forward_model.blocks.1.ffn.0", "forward_model.blocks.2.ffn.0", "forward_model.blocks.3.ffn.0",
    "motor_controller.confidence_head.0", "motor_controller.residual_blocks.0.0",
    "motor_controller.residual_blocks.1.0", "motor_controller.residual_blocks.2.0",
    "motor_controller.residual_blocks.3.0", "persistent_memory.outcome.0",
    "prediction_heads.heads.audio.mlp.0", "prediction_heads.heads.proprioception.mlp.0",
    "prediction_heads.heads.tactile.mlp.0", "prediction_heads.heads.vestibular.mlp.0",
    "prediction_heads.heads.vision.mlp.0", "recon_decoders.audio.0", "recon_decoders.proprioception.0",
    "recon_decoders.tactile.0", "recon_decoders.vestibular.0", "recon_decoders.vision.0",
    "self_state.grounded.local.0", "tenure_heads.order_proj.0", "tenure_heads.pose.0", "voice_head.mlp.0",
    "wellness.approximate.net.0", "wellness.approximate.net.2", "wellness.grounded.capability.net.0",
    "wellness.grounded.coherence.net.0", "wellness.grounded.motor_state.net.0",
    "wellness.grounded.prediction_error.net.0", "wellness.grounded.tactile_comfort.net.0")
PRODUCTION_POOLS = (
    "encoder_registry.encoders.motor.layers.0.attn", "encoder_registry.encoders.motor.layers.1.attn",
    "encoder_registry.encoders.motor.pool_attn", "encoder_registry.encoders.motor.token_read",
    "encoder_registry.encoders.tactile.layers.0.attn", "encoder_registry.encoders.tactile.layers.1.attn",
    "encoder_registry.encoders.tactile.pool_attn", "encoder_registry.encoders.tactile.token_read",
    "encoder_registry.encoders.vision.blocks.0.attn", "encoder_registry.encoders.vision.blocks.1.attn",
    "encoder_registry.encoders.vision.blocks.2.attn", "encoder_registry.encoders.vision.blocks.3.attn",
    "encoder_registry.encoders.vision.pool.attn",
    *[f"forward_model.blocks.{b}.{s}" for b in range(4) for s in ("cm_attn", "cmb_attn", "ct_attn")],
    "prediction_heads.heads.audio.attn", "prediction_heads.heads.proprioception.attn",
    "prediction_heads.heads.tactile.attn", "prediction_heads.heads.vestibular.attn",
    "prediction_heads.heads.vision.attn", "voice_head.attn")
PRODUCTION_BLOCKS = {
    "efference": ("body_mlp.0", "body_mlp.2", "gaze_in", "vision_in", "body_pool", "gaze_pool"),
    "contact_control": ("ctx1", "ctx2", "patch"),
    "synergy": ("E1", "E2", "S1", "S2"),
    "vestibular": ("encoder_registry.encoders.vestibular.mlp.0", "encoder_registry.encoders.vestibular.mlp.4",
                   "encoder_registry.encoders.vestibular.mlp.8")}
# the names the pages gave these parts before 2026-09-25 where the code names them otherwise, or where
# one kind of part had two names
RETIRED = ("250 ms pose head", "reflex controller", "reflex motor controller", "learned wellness copy, first",
           "memory's outcome layer", "movement controller's confidence", "first cross-sense attention",
           "first bridge attention", "motor encoder's first attention layer")
SOURCES = ("explainer/schema.py", "explainer/viewer.js", "explainer/walkthrough.py", "tools/build_explainer.py",
           "tools/development_dashboard.py", "tools/development_dashboard.js",
           "tools/development_dashboard_terms.js", "tools/lineage_series.py", "tools/dashboard.py")
# a dotted code path in a text, not a decimal number
CODE_PATH = re.compile(r"(?<![\w.])[a-z_]+(?:\.[a-z_0-9]+)+(?!\w)")


def _src(rel):
    return open(os.path.join(PKG, rel), encoding="utf-8").read()


def test_every_liveness_part_production_reports_has_a_name_and_a_definition():
    """No part of production's record prints as a code path on the Model health card, each report is
    named by its module, and each layer's kind is defined where the card names it. The control: a
    part the code does not have keeps its path, so the check can fail."""
    layers = [DD._response_name("model", k) for k in PRODUCTION_LAYERS]
    layers += [DD._response_name(b, k) for b, ks in PRODUCTION_BLOCKS.items() for k in ks if "pool" not in k]
    pools = [DD._part(k) for k in PRODUCTION_POOLS]
    pools += [DD._response_name(b, k) for b, ks in PRODUCTION_BLOCKS.items() for k in ks if "pool" in k]
    bare = [n for n in layers + pools if CODE_PATH.search(n) or "_" in n]
    assert not bare, bare
    assert DD._part("forward_model.blocks.0.not_a_layer") == "forward_model.blocks.0.not_a_layer"
    for b in PRODUCTION_BLOCKS:
        assert "_" not in DD._report(b), DD._report(b)
    assert DD._report("contact_control") == "contact head's report"
    assert DD._report("efference") == "efference heads' report"
    undefined = [n for n in layers if not DD._part_definitions([n])]
    assert not undefined, undefined
    # one name per kind: a block's first stage is named as its later ones are
    assert DD._part("forward_model.blocks.0.cm_attn") == "forward model, block 1: attention between senses"
    assert DD._part("encoder_registry.encoders.motor.layers.0.attn") == "motor encoder, attention layer 1"
    assert DD._part("recon_decoders.vestibular.0") == "the vestibular reconstruction decoder"


def test_the_history_charts_keep_the_code_path_beside_the_part_s_name_and_role():
    """Every code path in a history chart's title stands in parentheses right after the part's name,
    and the six known-dead parts' charts also say what the part does."""
    for key, spec in LS.SERIES.items():
        title = spec.get("title") or ""
        for m in CODE_PATH.finditer(title):
            path = m.group(0)
            name = DD._part(path)
            assert name != path and f"{name} ({path})" in title, (key, title)
    for part in LS.KNOWN_DEAD:
        name, role = LS.KNOWN_DEAD_WORDS[part]
        assert name == DD._part(part) and len(role.split()) >= 8, (part, role)
        assert LS.SERIES[f"live.{part}"]["title"] == f"Units responding: {name} ({part}): {role}"


def test_the_explainer_names_each_measured_part_beside_its_code_path():
    """The lineage table's columns, the newest sweep's bars and the popup rows, and the run tab's
    liveness chart lines: the dashboard's name with the code path beside it."""
    live = BX.read_liveness()
    sweep = json.load(open(os.path.join(BX.LIVENESS_DIR, "liveness_sweep_ckpt_step_344400.json")))
    row = next(r for r in live["rows"] if r["step"] == 344400)
    for c, v in zip(live["columns"], row["values"]):
        path = BX.LIVENESS_PARTS[c["key"]]
        assert c["path"] == path and c["name"] == f"{DD._part(path)} ({path})" and DD._part(path) != path
        # how each column was matched to its part: on the 344,400 row it reads the sweep's value
        want = (sweep["dead_layers"][path] if path in sweep["dead_layers"]
                else sweep["attention"][path]["entropy_share_of_uniform"])
        assert abs(float(v) - want) < 5e-4, (c["key"], v, want)
    sw = live["sweep"]
    shown = list(sw["dead_layers"]) + list(sw["locked_attention"])
    assert sorted(sw["names"]) == sorted(shown)
    for k in shown:
        assert DD._part(k) != k and sw["names"][k] == f"{DD._part(k)} ({k})", k
    js = _src("explainer/viewer.js")
    assert '[livenessName(k), v, ""]' in js and "[livenessName(k), isObj(a) ? a.top_share : a," in js
    assert 'livenessName(k) + ", in the liveness sweep "' in js and '"layer " + k' not in js
    recs = [{"type": "mimo", "step": s, "elapsed_s": s,
             "liveness": {"layers": {k: 0.5 for k in PRODUCTION_LAYERS[:3]},
                          "pools": {k: 0.5 for k in PRODUCTION_POOLS[:2]}}} for s in (100, 200, 300)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    labels = [s["label"] for c in out["charts"] if c.get("doc") == "chart.liveness" for s in c["series"]]
    for k in PRODUCTION_LAYERS[:3] + PRODUCTION_POOLS[:2]:
        assert f"{DD._part(k)} ({k})" in labels, (k, labels)


def test_the_retired_part_names_never_come_back():
    hits = [(rel, old) for rel in SOURCES for old in RETIRED if old in _src(rel)]
    assert not hits, hits


# one liveness reading per kind of part the Model health card defines (development_dashboard.
# PART_DEFINITIONS): the block that reports it and its key there
DEFINED = {
    "body head": ("model", "efference.body_mlp.0"), "gaze head": ("efference", "gaze_in"),
    "gaze copy": ("efference", "vision_in"), "prediction head": ("model", "prediction_heads.heads.vision.mlp.0"),
    "contact head": ("model", "contact_head.ctx1"), "voice head": ("model", "voice_head.mlp.0"),
    "wellness predictor": ("model", "wellness.approximate.net.0"),
    "evaluator's network": ("model", "wellness.grounded.coherence.net.0"),
    "tenure pose head": ("model", "tenure_heads.pose.0"),
    "tenure command projection": ("model", "tenure_heads.order_proj.0"),
    "reconstruction decoder": ("model", "recon_decoders.vision.0"),
    "motor controller": ("model", "motor_controller.residual_blocks.1.0"),
    "persistent memory's outcome head": ("model", "persistent_memory.outcome.0"),
    "feed-forward network": ("model", "forward_model.blocks.0.ffn.0"),
    "per-patch input layer": ("model", "encoder_registry.encoders.tactile.patch_mlp.0"),
    "per-body-part input layer": ("model", "encoder_registry.encoders.motor.patch_mlp.0"),
    "angular position encoding": ("model", "encoder_registry.encoders.vision.angular_pos_enc.mlp.0"),
    "self-state's grounded dynamics": ("model", "self_state.grounded.local.0"),
    "synergy map": ("synergy", "S1")}


def test_the_card_defines_each_kind_of_part_where_it_names_it(tmp_path):
    """One card per definition (review of 2026-09-25, finding 7: only the body head's line was held,
    and removing the gaze copy's passed): a part below the line is named in the card's issue, its
    kind's definition is in the card's explanation, and the issue carries it for the alert. A new
    definition needs a reading here."""
    assert set(DEFINED) == {key for key, _line in DD.PART_DEFINITIONS}
    for key, line in DD.PART_DEFINITIONS:
        block, name = DEFINED[key]
        record = ({"liveness": {"layers": {name: 0.3}}} if block == "model"
                  else {block: {"liveness": {name: 0.3}}})
        card = DD.liveness_card(record, measurements=tmp_path)
        # since 2026-09-30 (the operator: the model alert was "gobbledygook") the card's explanation
        # names only what the card shows; every kind's definition lives in the card's glossary (behind
        # its Definitions link) and in the flagged part's own issue, checked on the next line
        assert line in json.dumps(card["glossary"], ensure_ascii=False), (key, card["glossary"])
        assert card["issues"] and line in (card["issues"][0].get("definition") or ""), (key, card["issues"])

