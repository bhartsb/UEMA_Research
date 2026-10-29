# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The viewer generator: embedded data parses, seams are ALL seams, writes are
atomic and change-gated, and the page stays under its budget."""
from mimo_bridge.touch_adapter import PUSH_ARRIVAL_SCALE_CMS, PUSH_DEPTH_SCALE_CM
import json
import os
import re

from mimo_bridge.tools import build_explainer as BX


def test_learning_popups_follow_the_displayed_runs_training_and_replay_settings(tmp_path):
    """A training page must not inherit the frozen probe's learning/replay statements."""
    run = _fixture_run(tmp_path)
    args_path = os.path.join(run, "args.json")
    with open(args_path, encoding="utf-8") as stream:
        base = json.load(stream)
    for settings, update, replay in [
        ({"eval_only": True, "replay": "on"}, "model weights are frozen", "runs no replay updates"),
        ({"eval_only": False, "replay": "on"}, "training enabled", "replay updates are enabled"),
        ({"eval_only": False, "replay": "off"}, "training enabled", "no replay updates run"),
        ({}, "cannot identify whether weights were frozen", "cannot identify whether replay training was enabled"),
    ]:
        with open(args_path, "w", encoding="utf-8") as stream:
            json.dump(dict(base, **settings), stream)
        doc = _blocks(BX.render(run))["data-doc"]
        for key in ("bp.forward", "bp.grade", "bp.update"):
            assert update in doc[key]["w"], (settings, key)
            assert "H99_fixture" in doc[key]["w"]
        for key in ("bp.replay", "edge.replay.final_norm"):
            assert replay in doc[key]["w"], (settings, key)
            assert "H99_fixture" in doc[key]["w"]
        assert "every snapshot reports none" not in str(doc)
        assert "{tactile_comfort_definition}" not in str(doc)


def _fixture_touch_doc():
    """The column legend capture.py writes beside every touch row."""
    from mimo_bridge.explainer.capture import TOUCH_CLOUD_DOC
    return TOUCH_CLOUD_DOC


def _fixture_touch_rows():
    """Four touch rows at this build's real width: one pressed with a push, one the body
    pressing on ITSELF with no push, one holding only the vibration of a touch that has
    ended, and one unused slot. Written column by column from the code's own layout so a
    widened token fails here rather than shifting what the page draws."""
    from mimo_bridge.lem_model import TOUCH_TOKEN_F, TOUCH_VALID_COL
    rows = [[0.0] * TOUCH_TOKEN_F for _ in range(4)]
    doc = _fixture_touch_doc()
    at, col = {}, 0
    for raw in doc.strip().lstrip("[").rstrip("]").split(","):
        label = raw.strip()
        m = re.search(r"\bx(\d+)$", label)
        at[label] = col
        col += int(m.group(1)) if m else 1
    assert col == TOUCH_TOKEN_F, "the legend and the touch row disagree on the row's width"

    def put(row, label, *values):
        # the legend's own labels, matched on their opening words ("z (root-relative /0.5m)")
        start = next(c for lb, c in at.items() if lb == label or lb.startswith(label + " "))
        for i, v in enumerate(values):
            row[start + i] = v
    put(rows[0], "x", 0.20); put(rows[0], "y", 0.10); put(rows[0], "z", 0.05)
    put(rows[0], "pressure", 0.62); put(rows[0], "shear", 0.11)
    put(rows[0], "area", 0.30)
    put(rows[0], "push direction x3", 0.0, -1.0, 0.0)
    # the push's depth and arrival speed, on the scales the legend names (0.1 cm and 2 cm/s
    # since 2026-09-21): 0.15 cm deep and arriving at 1.2 cm/s, both inside what the crib body
    # measures, whose deepest contact was 0.312 cm and fastest arrival 14.0 cm/s
    put(rows[0], "depth", 0.15 / PUSH_DEPTH_SCALE_CM)
    put(rows[0], "arrival speed", 1.2 / PUSH_ARRIVAL_SCALE_CMS)
    put(rows[1], "x", -0.10); put(rows[1], "y", 0.20)
    put(rows[1], "pressure", 0.31); put(rows[1], "area", 0.10)
    put(rows[1], "self-touch share", 0.80)
    put(rows[1], "own parts on it /4", 0.25)
    put(rows[2], "x", 0.02); put(rows[2], "y", -0.30); put(rows[2], "z", 0.10)
    put(rows[2], "vib x4", 0.10, 0.05, 0.0, 0.0)    # the touch ended within the last second
    for r in rows[:3]:
        r[TOUCH_VALID_COL] = 1.0
    return rows


def _fixture_entity_self():
    """One stats() shape from mimo_bridge/entity_self.py, both levels, with the minutes
    level still short of the windows a link needs (so its count reads -1) - the state every
    real run is in today."""
    names = ["caregiver", "toy_ball", "mobile"]
    return {"frames": 76333, "entities": len(names), "teacher": "wellness",
            "null_mode": "level", "bodies": 2, "names": names,
            "presence_mean": [0.82883, 0.19375, 0.5888], "windows_needed": 11,
            "l2_ticks": 6, "l2_window_moments": 10000, "l2_entity_windows": [6, 6, 5],
            "l2_link": [0.0647, 0.0149, 0.0583], "l2_rank": [0.6593, 0.5611, 0.5567],
            "l2_line": None, "l2_counted": -1, "l2_counted_names": [],
            "l2_entities_old_enough": 0, "l2_steps_to_first_evidence": 50000,
            "l3_ticks": 0, "l3_window_moments": 100000, "l3_entity_windows": [0, 0, 0],
            "l3_link": [0.0088, -0.0295, 0.0057], "l3_rank": [0.5, 0.5, 0.5],
            "l3_line": None, "l3_counted": -1, "l3_counted_names": [],
            "l3_entities_old_enough": 0, "l3_steps_to_first_evidence": 1100000,
            "reader": {"weight": 0.1, "level": "l2", "gate_open": False, "gate_streak": 0,
                       "gate_windows_needed": 3, "counted_entities": [], "n_counted": 0,
                       "moments_compared": 27664, "applied_weight": 0.0,
                       "real_tracks_wellness": 0.0, "control_tracks_wellness": 0.0,
                       "why": "no entity has passed the evidence test yet, so the reader "
                              "has nothing to read"}}


def _fixture_run(tmp_path, n_snap=3, n_rec=30, restarts=(12,)):
    run = tmp_path / "H99_fixture"
    (run / "explain").mkdir(parents=True)
    (run / "checkpoints" / "ckpt_step_100").mkdir(parents=True)
    json.dump({"scene": "crib", "n_envs": 2, "context_length": 4, "motor_hold": 5,
               "residual_prediction": "on", "next_frame_loss": "on",
               "replay_every": 4, "replay_batch": 48, "action_repeat": 5},
              open(run / "args.json", "w"))
    handle_names = (["wrist_L", "thumb_L", "index_L", "middle_L", "ring_L", "pinky_L",
                     "wrist_R", "thumb_R", "index_R", "middle_R", "ring_R", "pinky_R",
                     "head", "upperarm_L", "lowerarm_L", "upperarm_R", "lowerarm_R",
                     "thigh_L", "calf_L", "thigh_R", "calf_R", "foot_L", "foot_R",
                     "gaze"] + ["lower_trunk", "chest"])   # trunk body: 26, gaze at 23
    # The learned-weight presentation - bold boxes, the count at each corner, the
    # whole footer line - renders only when the receipt carries per-component counts,
    # and the core's own boxes size themselves from the per-tensor inventory. The
    # fixture carried neither, so none of that code was reachable by any test.
    fm_tensors = {
        "modality_embedding.weight": {"shape": [6, 128], "kind": "parameter", "elements": 768},
        "temporal_embedding.weight": {"shape": [4, 128], "kind": "parameter", "elements": 512},
        "final_norm.weight": {"shape": [128], "kind": "parameter", "elements": 128},
        "final_norm.bias": {"shape": [128], "kind": "parameter", "elements": 128}}
    for blk in (0, 1):
        fm_tensors[f"blocks.{blk}.causal_mask"] = {"shape": [4, 4], "kind": "buffer", "elements": 16}
        for stage in ("cm", "ct", "cmb"):
            fm_tensors[f"blocks.{blk}.{stage}_norm.weight"] = {
                "shape": [128], "kind": "parameter", "elements": 128}
            fm_tensors[f"blocks.{blk}.{stage}_attn.in_proj_weight"] = {
                "shape": [384, 128], "kind": "parameter", "elements": 49152}
        fm_tensors[f"blocks.{blk}.ffn_norm.weight"] = {
            "shape": [128], "kind": "parameter", "elements": 128}
        fm_tensors[f"blocks.{blk}.ffn.0.weight"] = {
            "shape": [512, 128], "kind": "parameter", "elements": 65536}
    # THIS FIXTURE's core, not any run's: 2 blocks at token width 128 with a 4-moment
    # context, and only the tensors listed above (no biases, no out-projections), which
    # makes 213,504 learned numbers per block; 428,544 for the whole core with the tags and
    # the final norm, and 32 more elements saved as the two blocks' fixed masks. A real
    # 128-wide block holds 330,880 learned numbers and H84's 192-wide block 742,080 (its
    # receipt's inventory); _fixture_run_at_shape rebuilds this inventory at another width
    # and block count
    component_params = {"enc_vision": 1000000, "enc_tactile": 400000,
                        "enc_audio": 200000, "enc_proprioception": 100000,
                        "enc_motor": 200000, "enc_intermind": 100000,
                        "forward_model": 428544, "prediction_heads": 300000,
                        "recon_decoders": 100000, "motor_controller": 150000,
                        "wellness": 50000, "gaze_head": 83202, "td_gates": 15,
                        "tenure_heads": 4037}
    json.dump({"active_modalities": ["vision", "tactile", "audio", "proprioception",
                                    "motor", "intermind"],
               "encoders_built": {m: {"type": "mlp_project", "d_output": 128}
                                  for m in ("vision", "tactile", "audio",
                                            "proprioception", "motor", "intermind")},
               "forward_model": {"_seq_length": 24, "_n_modalities": 6,
                                 "n_layers": 2, "n_heads": 8},
               "handle_names": handle_names,
               "motor_controller": {"output_per_handle": 9},
               # without this the (final_norm, voice) edge rule and the whole voice
               # slide were unreachable: no test ever ran that branch. 93,264 is the
               # count H81's receipt carries for a d_model of 128.
               "voice_head": {"enabled": True, "output_dim": 80, "params": 93264},
               "component_params": component_params,
               "checkpoint_inventory": {
                   "schema": "saved_module_inventory_v1", "step": 100,
                   "optimizer_parameters": sum(component_params.values()),
                   "optimizer_parameter_tensors": 42,
                   "components": {
                       "forward_model": {"parameters": 428544,
                                         "saved_tensor_elements": 428576,
                                         "tensors": fm_tensors},
                       "gaze_head": {"parameters": 83202, "saved_tensor_elements": 83206,
                                     "tensors": {}}}},
               "total_params": sum(component_params.values())},
              open(run / "checkpoints" / "ckpt_step_100" / "model_receipt.json", "w"))
    with open(run / "replay_training_stats.jsonl", "w") as fh:
        fh.write(json.dumps({"type": "launch", "step": 0, "timestamp": 0}) + "\n")
        elapsed = 0.0
        for i in range(n_rec):
            step = i * 100
            if step in restarts:      # a restart: elapsed_s drops
                elapsed = 5.0
                fh.write(json.dumps({"type": "launch", "step": step,
                                     "timestamp": step}) + "\n")
            elapsed += 60.0
            # every field some chart reads, so all of them render and every chart
            # doc entry has a chart to open from
            fh.write(json.dumps({
                "type": "mimo", "step": step, "elapsed_s": elapsed,
                "causal_pe": 0.1 / (i + 1),
                "causal_pe_per_modality": {"vision": 0.2 / (i + 1)},
                "pe": {"vision": 0.1},
                "wellness": {"coherence": 0.5, "prediction_error": 0.4,
                             "tactile_comfort": 0.6, "capability": 0.3,
                             "motor_state": 0.5, "movement": 0.2},
                "movement": {"local_speed": 3.0 + i, "stillness_debt": 0.1},
                "tenure_tok": {"rank1": 0.5, "vis_gate_open": 1.0, "n_decoys": 7},
                "tenure_gap": {"pose": 0.01}, "tenure_pe": {"pose": 0.05},
                "recon_dec": {"vision": 0.3},
                "agency": {"pay": 0.2, "prior_w": 0.05, "vis_adv": 0.1,
                           "touch_adv": 0.2},
                "td_gate": {"vision": 0.0, "tactile": 0.1},
                "contract": {"exec_seen": i, "step": step},
                "vision_fresh_fraction": 1.0,
                "token_temporal_std": {"vision": 0.2},
                "curiosity_drive": 0.4, "cmd_mag_rad": 0.5,
                "cmd_response_cos": 0.3, "cmd_response_cos_held": 0.7,
                "replay_loss": 0.5 / (i + 1),
                "model_steps_per_s": 7.0}) + "\n")
            fh.write(json.dumps({"type": "training", "step": step,
                                 "avg_loss": 1.0 / (i + 1),
                                 "loss_breakdown": {"next_frame": 0.5}}) + "\n")
        fh.write(json.dumps({"type": "phase_transition", "step": 900,
                             "phase": "early_infant"}) + "\n")
    entries = []
    for k in range(n_snap):
        step = k * 1000
        snap = {"header": {"schema_version": 1, "run": "H99_fixture", "step": step,
                           "timestamp": 0, "phase": "neonate", "source_mask": {"vision": "real"},
                           "n_envs": 2, "body": 0, "weights": "frozen", "capture_ms": 20,
                           "dev_limits": {}, "flags": {}},
                "model": {}, "tokens": {"raw": {}, "token_temporal_std": {}},
                "predictions": {"of_now": "absent: fixture", "error_of_now": {},
                                "recon_dec_error": {}, "of_next": "absent: fixture",
                                "causal_pe": 0.1, "recon_pe": 0.2,
                                "actual_now_is": "tokens.raw"},
                "attention": {"modalities": [], "ctx": 4, "seq": 24,
                              "heads": "averaged", "inputs": "fixture", "layers": []},
                # the touch rows are this build's real width (lem_model.TOUCH_TOKEN_F), with
                # the push columns filled on one row and empty on another, so the panel that
                # draws them is reachable by the headless-browser check
                "senses": {"eyes": "absent: fixture",
                           "touch_cloud": _fixture_touch_rows(),
                           "touch_cloud_doc": _fixture_touch_doc(),
                           "contacts": 2, "task": {}},
                "motor": {"handle_names": [], "motor_out": [], "motor_for_action": [],
                          "prev_executed": "absent", "held_order": "absent",
                          "order_seq": 1, "order_age": 0, "gaze_az_el": None,
                          "confidence": 0.5, "wellness_grad_norm": 0.0},
                "drives": {"wellness_body": {"prediction_error": 0.91,
                                             "tactile_comfort": 0.50, "capability": 0.83,
                                             "coherence": 0.27, "motor_state": 0.96,
                                             "self_touch": 0.0},
                           "wellness_batch_mean": {},
                           "coherence_source": "body_schema_extent",
                           "curiosity_drive": 0.5, "curiosity_zone": "goldilocks",
                           "movement": None, "agency": None, "td_gate": None,
                           "tenure": {"pe": {}, "gap": {}, "tok": {}}},
                # UEMA 11.2 and the named-entity levels, so both panels are reachable by
                # the headless-browser check; the toy is a mocap body here as in the world
                "incorporation": {
                    "handles_mine": 0.28235, "extent": 0.26943, "certainty": 0.4353,
                    "carried_by_wellness": "extent",
                    "objects": {"toy": 0.11816, "mobile": 0.08461},
                    "objects_per_body": {"toy": [0.07586, 0.16210],
                                         "mobile": [0.07586, 0.13568]},
                    "objects_evidence": {
                        "toy": {"match": 0.31, "control": 0.19, "earned": 0.12},
                        "mobile": {"match": 0.10, "control": 0.12, "earned": 0.0}},
                    "objects_physics_driven": {"toy": False, "mobile": True}},
                "entity_self": _fixture_entity_self(),
                "intention": {"last": None, "goal_ring_size": 0,
                              "best_remembered_wellness": None, "scale_report": None},
                "losses": {"loss": 1.0, "breakdown": {}, "grad_norm": 0.1,
                           "imagination_loss": None, "scope": "live", "replay": None,
                           "timings_ms": {}}}
        name = f"step_{step}.json"
        json.dump(snap, open(run / "explain" / name, "w"))
        entries.append({"step": step, "file": name, "bytes": 100, "mtime": 0})
    json.dump({"snapshots": entries}, open(run / "explain" / "index.json", "w"))
    return str(run)


def _blocks(page):
    out = {}
    for m in re.finditer(r'<script type="application/json" id="([^"]+)">(.*?)</script>',
                         page, re.S):
        out[m.group(1)] = json.loads(m.group(2).replace("<\\/", "</"))
    return out


def test_page_builds_and_embedded_json_parses(tmp_path):
    run = _fixture_run(tmp_path)
    page = BX.render(run)
    for tab in ("tab-arch", "tab-inspect", "tab-run"):
        assert tab in page
    b = _blocks(page)
    for bid in ("data-meta", "data-snapshots", "data-series", "data-tour", "data-doc"):
        assert bid in b, f"missing embedded block {bid}"
    assert b["data-meta"]["run"] == "H99_fixture"
    assert len(b["data-snapshots"]) == 3
    assert len(b["data-tour"]) >= 14
    assert ">None<" not in page
    assert len(page.encode()) < BX.PAGE_BUDGET_MB * 1e6


def test_seams_are_all_seams_and_markers_present(tmp_path):
    run = _fixture_run(tmp_path, restarts=(400, 1200))
    b = _blocks(BX.render(run))
    s = b["data-series"]
    assert s["seams"] == [400, 1200], "every restart, not just the last segment"
    assert s["phases"] and s["phases"][0]["step"] == 900
    assert s["snapmarks"] == [0, 1000, 2000]
    labels = [c["label"] for c in s["charts"]]
    assert any("command explain the outcome" in l for l in labels), "rank1 chart is load-bearing"
    assert any("Body speed" in l for l in labels)
    assert any("Replay memory" in l for l in labels)
    # importance order: the rank1 chart comes first, and it carries a trend spec
    assert "command explain the outcome" in s["charts"][0]["label"]
    assert s["charts"][0]["trend"] == {"key": "tenure_tok.rank1", "dir": "higher"}
    # clarity is load-bearing: every chart ships a plain-language caption, and
    # every series carries its key (the verdict tiles look series up by key)
    assert all(c.get("caption") for c in s["charts"]), "a chart without a caption"
    assert all(sr.get("key") for c in s["charts"] for sr in c["series"])
    # the exec-contract fraction resets at each segment's launch record
    fresh = next(c for c in s["charts"] if "executed command reach" in c["label"])
    exec_series = next(x for x in fresh["series"] if x["key"] == "exec_frac")
    assert all(0.0 <= p[1] <= 1.0 for p in exec_series["points"])


def test_writes_are_atomic_and_change_gated(tmp_path, monkeypatch):
    # every build reads the clock a second later than the one before: the page's build time
    # (generated_at, stamped to the second) is not a change of content (review of 2026-09-25,
    # finding 13: two builds that straddled a second were written twice)
    real, tick = BX.time.strftime, [0]

    def later(fmt, *rest):
        if fmt == "%Y-%m-%d %H:%M:%S" and not rest:
            tick[0] += 1
            return "2026-09-25 19:00:%02d" % tick[0]
        return real(fmt, *rest)
    monkeypatch.setattr(BX.time, "strftime", later)
    run = _fixture_run(tmp_path)
    out2 = str(tmp_path / "copy.html")
    written = BX.build(run, out2, 12, False, 30.0)
    assert sorted(os.path.basename(p) for p in written) == ["copy.html", "explainer.html"]
    assert not os.path.exists(os.path.join(run, "explainer.html.tmp"))
    first = open(out2, encoding="utf-8").read()
    # identical content a second later -> no write at all; the page keeps its own build time
    again = BX.build(run, out2, 12, False, 30.0)
    assert again == [] and tick[0] >= 2
    assert open(out2, encoding="utf-8").read() == first
    # changed stats -> rebuilt
    with open(os.path.join(run, "replay_training_stats.jsonl"), "a") as fh:
        fh.write(json.dumps({"type": "mimo", "step": 99999, "elapsed_s": 1,
                             "causal_pe": 0.01}) + "\n")
    assert BX.build(run, out2, 12, False, 30.0)


def test_snapshot_thinning_keeps_newest(tmp_path):
    run = _fixture_run(tmp_path, n_snap=9)
    snaps, all_steps = BX.pick_snapshots(run, 4)
    assert len(snaps) == 4 and len(all_steps) == 9
    assert snaps[-1]["header"]["step"] == 8000, "the newest snapshot always survives"


def _drawable_edges(receipt, args):
    """Mirror of viewer.js buildDiagram: every (from, to) line it draws. A tour
    edge outside this set would silently animate nothing. Hand-written, and held to
    viewer.js in both directions by
    test_the_edge_mirror_and_viewer_js_name_the_same_lines."""
    mods = receipt.get("active_modalities", [])
    nl = int(receipt.get("forward_model", {}).get("n_layers", 0) or 2)
    E = set()
    for m in mods:
        E |= {("world", f"sense_{m}"), (f"sense_{m}", f"enc_{m}"), (f"enc_{m}", "ln_out")}
    # the gate box stands between the shared LayerNorm and the assembly, so with the
    # gate on the one line becomes two and the direct line is not drawn at all; no
    # single run can draw both shapes, which is why callers that want every line the
    # diagram is ABLE to draw union this over both settings
    if (args.get("top_down_gate") or "off") != "off":
        E |= {("ln_out", "td_gate"), ("td_gate", "assembly")}
    else:
        E.add(("ln_out", "assembly"))
    E.add(("assembly", "block0_s1"))
    for i in range(nl):
        E |= {(f"block{i}_s1", f"block{i}_s2"), (f"block{i}_s2", f"block{i}_s3"),
              (f"block{i}_s3", f"block{i}_ffn")}
        if i:
            E.add((f"block{i - 1}_ffn", f"block{i}_s1"))
    E.add((f"block{nl - 1}_ffn", "final_norm"))
    for t in ("heads", "recon", "motor"):
        E.add(("final_norm", t))
    # the delayed re-run's prediction of this model step's tokens (lem_model.py self._nf_pred) is read by the
    # top-down gate, by wellness (prediction error, learning progress) and by the curiosity drive (the code check
    # of Oct 15, 2026)
    E |= {("heads", "wellness"), ("heads", "curiosity")}
    if (args.get("top_down_gate") or "off") != "off":
        E.add(("heads", "td_gate"))
    # the learned wellness scorer reads the RAW ENCODER tokens (lem_model.py passes
    # canonical_tokens = the shared-LayerNorm tokens), so its line starts at ln_out;
    # the wellness slope reaches the motor controller on a line of its own
    E |= {("ln_out", "wellness"), ("wellness", "motor")}
    # the curiosity drive multiplies the felt value stored with each goal-ring moment
    # inside the intention block, so its line ends at the planner and not at wellness;
    # the curriculum's two outputs land in different places - the phase limits clamp
    # every order the body receives, a phase change is a message to the world workers
    E |= {("motor", "gaze"), ("motor", "world"), ("gaze", "world"), ("intention", "motor"),
          ("wellness", "intention"), ("curiosity", "intention"), ("curriculum", "motor")}
    # a run that starts in the final phase with its age fixed and its delays off can never change phase, so
    # it draws no phase-change line to the world (train_mimo's defaults: --age-binding curriculum,
    # --delay-binding on)
    if not (args.get("start_phase") == "mature" and str(args.get("age_binding") or "").startswith("fixed")
            and (args.get("delay_binding") or "on") == "off"):
        E.add(("curriculum", "world"))
    # the second learning channel: every moment writes a row into the ring and one
    # extra optimizer step comes back out of it into the same weights
    E |= {("ln_out", "replay"), ("replay", "final_norm")}
    if (args.get("imagination") or "config") != "off":
        E |= {("ln_out", "imagination"), ("motor", "imagination")}
        # the chain's win rate sets the planner's depth only under --hand-target-mode world; otherwise the
        # search looks one model step ahead (lem_model.py _plan_chain)
        if (args.get("hand_target_mode") or "world") == "world":
            E.add(("imagination", "intention"))
    if (receipt.get("persistent_memory") or args.get("persistent_memory")) == "on":
        E |= {("memory", "final_norm"), ("ln_out", "memory")}
    if (receipt.get("self_state") or args.get("self_state") or "off") != "off":
        E.add(("ln_out", "self_state"))
    # the caregiver runs inside the world's own physics step: what it reads and what it changes
    if (args.get("caregiver_mode") or "off") != "off":
        E |= {("world", "caregiver"), ("caregiver", "world")}
    if (receipt.get("voice_head") or {}).get("enabled"):
        E.add(("final_norm", "voice"))
    if (args.get("hierarchical_self") or "off") != "off":
        # the observer reads this step's wellness AND this step's raw tokens
        E |= {("wellness", "hself"), ("ln_out", "hself")}
    # the gain-of-function build (2026-09-22): the named things and their reader, and the
    # side heads the receipt says were built - each handed the order and the core's state
    if (args.get("entity_self") or "off") != "off":
        E |= {("wellness", "entity_self"), ("entity_self", "intention"), ("world", "entity_self")}
    if receipt.get("efference_heads"):
        E |= {("motor", "efference"), ("final_norm", "efference")}
        # the gaze copy (lem/model/efference.py GazeCopy) adds its output to the prediction heads'
        # vision token; the heads' stamp records gaze_copy_into_vision whenever the model has
        # vision, and viewer.js draws the line on that stamp alone
        eff = receipt["efference_heads"]
        if isinstance(eff, dict) and eff.get("gaze_copy_into_vision"):
            E.add(("efference", "heads"))
        # where the heads reach the planner (schema.efference_routes, mirrored by hand here): the
        # goal ranking's controllability and learning-progress channels under --wellness-rank
        # normalized, at a weight other than 0 (the receipt's wellness_channels stamp wins over the
        # launch flags), and the action search under --physical-goals memory with --intention on
        wc = receipt.get("wellness_channels") or {}
        if (wc.get("rank") or args.get("wellness_rank") or "raw") == "normalized":
            weights = {"controllability": 1.0, "learning_progress": 0.75}
            weights.update(wc.get("weights") or {})
            ranked = wc.get("channels") or None
            if any((c in ranked if ranked else args.get(c) == "on") and weights[c] != 0
                   for c in ("controllability", "learning_progress")):
                E.add(("efference", "wellness"))
        if args.get("physical_goals") == "memory" and args.get("intention") == "on":
            E.add(("efference", "intention"))
    if receipt.get("contact_head"):
        E |= {("motor", "contact_head"), ("final_norm", "contact_head"), ("contact_head", "wellness")}
    if receipt.get("motor_synergy"):
        E |= {("world", "synergy_examples"), ("synergy_examples", "synergy")}
        if receipt["motor_synergy"].get("planner") is not False:
            E.add(("synergy", "intention"))
        if receipt["motor_synergy"].get("babble"):
            E.add(("synergy", "world"))
    # THE INPUT CHECK OF OCT 15, 2026 (docs/explainer_inputs_2026-10-15): the inputs the drawing lacked.
    # The tactile and the motor sense's raw readings, read without the encoder: wellness's skin-comfort, self-contact,
    # handle-position and capability scores, the motor controller's handle state, persistent memory's measured needs
    # and self perception's per-site skin fields
    mem_on = (receipt.get("persistent_memory") or args.get("persistent_memory")) == "on"
    self_on = (receipt.get("self_state") or args.get("self_state") or "off") != "off"
    gate_on = (args.get("top_down_gate") or "off") != "off"
    if "tactile" in mods:
        E.add(("sense_tactile", "wellness"))
        if mem_on:
            E.add(("sense_tactile", "memory"))
        if self_on:
            E.add(("sense_tactile", "self_state"))
    if "motor" in mods:
        E |= {("sense_motor", "motor"), ("sense_motor", "wellness"), ("motor", "enc_motor"), ("sense_motor", "replay")}
        if mem_on:
            E.add(("sense_motor", "memory"))
        # the independent review of Oct 15, 2026 (S1): the goal ring's stored pose under either memory goal flag
        if args.get("hand_goals") == "memory" or args.get("physical_goals") == "memory":
            E.add(("sense_motor", "intention"))
    # the planner's starting command (S1), the replay row's executed command (m8), and the world's consequence count
    # under --consequence-w above 0 (m12)
    E |= {("motor", "intention"), ("motor", "replay")}
    if float(args.get("consequence_w") or 0) > 0:
        E.add(("world", "intention"))
    # the executed command into persistent memory and self perception; the world's time stamp and readings; the
    # planner's last goal; persistent memory's state into the replay ring; the core's output into self perception
    if mem_on:
        E |= {("motor", "memory"), ("world", "memory"), ("intention", "memory"), ("memory", "replay")}
    if self_on:
        E |= {("motor", "self_state"), ("world", "self_state"), ("final_norm", "self_state")}
    # the world's contact report into the contact head
    if receipt.get("contact_head"):
        E.add(("world", "contact_head"))
    # the tokens the goal ring stores and the planner searches from: the top-down gate's output, or the shared
    # LayerNorm's with the gate off; with the gate on the imagination chain starts from the same set
    if gate_on:
        E.add(("td_gate", "intention"))
        if (args.get("imagination") or "config") != "off":
            E.add(("td_gate", "imagination"))
    else:
        E.add(("ln_out", "intention"))
    # --residual-prediction on: each head's output is added to its sense's raw token
    if (args.get("residual_prediction") or "off") == "on":
        E.add(("ln_out", "heads"))
    return E


def test_tour_edges_are_drawable_and_the_loop_slide_traces_the_loop(tmp_path):
    run = _fixture_run(tmp_path)
    b = _blocks(BX.render(run))
    receipt = b["data-meta"]["receipt"]
    args = json.load(open(os.path.join(run, "args.json")))
    drawable = _drawable_edges(receipt, args)
    tour = b["data-tour"]
    for st in tour:
        assert "edges" in st, f"step {st['id']} declares no edges field"
        for a, to in st["edges"]:
            assert (a, to) in drawable, \
                f"step {st['id']} animates a line the diagram never draws: {a}->{to}"
    # the loop slide traces the WHOLE circuit - in through every sense, through
    # both blocks with no dark gap, out through heads and motor, back to the world
    loop = {tuple(e) for e in tour[0]["edges"]}
    for m in receipt["active_modalities"]:
        assert ("world", f"sense_{m}") in loop, f"loop slide skips sense {m}"
    for need in [("ln_out", "assembly"), ("assembly", "block0_s1"),
                 ("block0_ffn", "block1_s1"), ("block1_ffn", "final_norm"),
                 ("final_norm", "heads"), ("final_norm", "motor"), ("motor", "world")]:
        assert need in loop, f"loop slide is missing {need}"
    # no slide animates the side paths the old touching rule lit during the loop
    assert ("memory", "final_norm") not in loop
    assert ("intention", "motor") not in loop


def test_the_top_down_gate_is_drawn_when_the_run_had_it_on(tmp_path):
    """With --top-down-gate on, the tokens the core reads are not the encoders'
    output, so the page must show the gate as its own station and the tour's line
    must run through it rather than straight from the norm to the assembly."""
    run = _fixture_run(tmp_path)
    args_path = os.path.join(run, "args.json")
    args = json.load(open(args_path))
    args["top_down_gate"] = "on"
    json.dump(args, open(args_path, "w"))
    b = _blocks(BX.render(run))
    drawable = _drawable_edges(b["data-meta"]["receipt"], args)
    assert ("ln_out", "td_gate") in drawable and ("td_gate", "assembly") in drawable
    assert ("ln_out", "assembly") not in drawable
    loop = {tuple(e) for e in b["data-tour"][0]["edges"]}
    assert ("ln_out", "td_gate") in loop and ("td_gate", "assembly") in loop
    assert ("ln_out", "assembly") not in loop
    for st in b["data-tour"]:
        for a, to in st["edges"]:
            assert (a, to) in drawable, \
                f"step {st['id']} animates a line the diagram never draws: {a}->{to}"


def _viewer_js_edges(src, mods, n_layers):
    """Every (from, to) line buildDiagram can draw, read out of viewer.js itself.

    A call whose two endpoints are string literals is taken as written. A call that
    builds an endpoint from an expression is expanded by the table below, keyed on the
    EXACT source text of its two endpoint expressions - so re-routing such a call
    changes its text, the lookup misses, and the test fails naming the call."""
    seg = src[src.index("function buildDiagram()"):src.index("function buildLearnLoop()")]
    # the call may carry the comb's name after the box (edge(a, b, cls, bus), Oct 15, 2026)
    outs = re.search(r'\[((?:"[A-Za-z0-9_]+"(?:,\s*)?)+)\]\.forEach\(t => edge\("final_norm", t(?:,[^()]*)?\)\)', seg)
    assert outs, "the final_norm output list moved - this check stopped reading it"
    out_boxes = re.findall(r'"([A-Za-z0-9_]+)"', outs.group(1))
    L = range(n_layers)
    expand = {
        ('"sense_" + m', '"enc_" + m'): {(f"sense_{m}", f"enc_{m}") for m in mods},
        ('"world"', '"sense_" + m'): {("world", f"sense_{m}") for m in mods},
        ('"enc_" + m', '"ln_out"'): {(f"enc_{m}", "ln_out") for m in mods},
        # one expression draws every link inside a block since Oct 7, 2026 (the decided design of the Architecture
        # tab, D-E5): block 0's boxes and the later blocks' cells walk the same list of parts, the FFN its last
        ('`block${i}_${prev}`', '`block${i}_${sid}`'):
            {(f"block{i}_s1", f"block{i}_s2") for i in L} | {(f"block{i}_s2", f"block{i}_s3") for i in L}
            | {(f"block{i}_s3", f"block{i}_ffn") for i in L},
        ('"assembly"', '`block0_${firstBox}`'): {("assembly", "block0_s1")},
        ('`block${i - 1}_ffn`', '`block${i}_${firstBox}`'):
            {(f"block{i - 1}_ffn", f"block{i}_s1") for i in L if i},
        ('`block${nL - 1}_ffn`', '"final_norm"'): {(f"block{n_layers - 1}_ffn", "final_norm")},
        ('"final_norm"', 't'): {("final_norm", t) for t in out_boxes},
    }
    drawn, unread = set(), []
    for m in re.finditer(r'\b(?:routeSolid|edge|route|draw)\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*[,)]', seg):
        a, b = m.group(1), m.group(2)
        if (a, b) == ("a", "b"):                       # the four helpers' own bodies
            continue
        la, lb = re.fullmatch(r'"([A-Za-z0-9_]+)"', a), re.fullmatch(r'"([A-Za-z0-9_]+)"', b)
        if la and lb:
            drawn.add((la.group(1), lb.group(1)))
        elif (a, b) in expand:
            drawn |= expand[(a, b)]
        else:
            unread.append((a, b))
    assert not unread, (
        f"buildDiagram draws lines from endpoint expressions this check cannot expand: "
        f"{unread} - add each to the table here AND to _drawable_edges")
    return drawn


def test_the_edge_mirror_and_viewer_js_name_the_same_lines():
    """Audit P195. _drawable_edges is hand-written, and every tour edge is checked
    against it, so it has to be held to viewer.js in BOTH directions: a line viewer.js
    draws that the mirror lacks, and a line the mirror lists that viewer.js no longer
    draws - a tour slide naming one of those animates nothing. The earlier form of this
    test read only the 21 calls whose two endpoints are string literals and checked one
    direction; measured 2026-09-19, deleting the memory line and re-routing the
    encoder, core-entry and output lines all passed it."""
    src = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    mods = ["vision", "tactile", "audio", "proprioception", "motor", "intermind"]
    # every stamp the diagram draws a line for, so the mirror names every line viewer.js CAN draw:
    # gaze_copy_into_vision is the efference stamp's record of the gaze copy (the line
    # efference -> heads, since commit aebef83c1)
    receipt = {"active_modalities": mods, "forward_model": {"n_layers": 2},
               "persistent_memory": "on", "voice_head": {"enabled": True},
               "efference_heads": {"weight": 0.1, "gaze_copy_into_vision": True},
               "contact_head": {"weight": 0.1},
               "motor_synergy": {"dim": 16, "babble": True}}
    mirror = set()
    for gate in ("off", "on"):            # the two gate settings draw different lines
        # the launch settings under which the efference heads reach the planner, so the two lines
        # out of the efference box are listed too (schema.efference_routes)
        mirror |= _drawable_edges(receipt, {"hierarchical_self": "dark", "top_down_gate": gate,
                                            "entity_self": "dark", "wellness_rank": "normalized",
                                            "controllability": "on", "physical_goals": "memory",
                                            "intention": "on", "caregiver_mode": "on",
                                            "self_state": "learn", "residual_prediction": "on",
                                            "consequence_w": 0.5})
    drawn = _viewer_js_edges(src, mods, 2)
    assert len(drawn) >= 51, f"only {len(drawn)} lines read out of buildDiagram - 51 on 2026-09-19"
    only_js = sorted(drawn - mirror)
    only_mirror = sorted(mirror - drawn)
    assert not only_js, f"viewer.js draws lines _drawable_edges never lists: {only_js}"
    assert not only_mirror, (
        f"_drawable_edges lists lines viewer.js no longer draws, so a tour slide naming "
        f"one would animate nothing: {only_mirror}")


def _fixture_run_gaze_copy(tmp_path, physical_goals=None):
    """_fixture_run with version 2 efference heads, whose stamp records the gaze copy (what
    lem/model/efference.py stamp() writes for a model with vision), launched with
    --physical-goals set to `physical_goals` (None: a launch from before the flag existed)."""
    run = _fixture_run(tmp_path)
    _set_args(run, efference_heads="on", efference_weight=0.1,
              **({} if physical_goals is None else {"physical_goals": physical_goals}))

    def rec_edit(r):
        r["component_params"]["efference_heads"] = 500000
        r["total_params"] = sum(r["component_params"].values())
        r["checkpoint_inventory"]["optimizer_parameters"] = r["total_params"]
        r["efference_heads"] = {"version": "efference_v2", "weight": 0.1, "tenure_steps": 5,
                                "scales": {"pos_next_cm": 0.452, "rot_next_deg": 2.73},
                                "gaze_copy_into_vision": True, "gaze_head_repurposed": False}
    _edit_receipt(run, rec_edit)
    return run


def test_the_efference_popups_are_the_maps_own_and_say_what_the_code_does(tmp_path):
    """2026-09-25. Commit aebef83c1 drew the gaze copy's line (efference -> heads) and wrote
    node.efference and edge.efference.heads into DOC from viewer.js at load: the page opened 279
    popups while the map the suite checks held 278, and the rewritten node.efference said on every
    page that the body head has no planner connection. The two entries live in schema.py now, and
    what they say is held to the code here: the gaze copy joins the vision prediction inside the
    one transition every imagined step uses, the line is drawn on the stamp the heads write from
    their vision index, and the planner's action search reads the body head only through a
    physical goal, which the model stores only under --physical-goals memory."""
    import inspect
    from lem.model import efference as EF
    from lem.model import predictive_state as PS
    from mimo_bridge import lem_model as LM
    assert not re.search(r"\bDOC\s*\[[^\]]+\]\s*=(?!=)", _viewer_js()), \
        "viewer.js writes a popup entry at load: its text belongs in schema.py DOC"
    doc = _page_doc()
    assert "edge.efference.heads" in doc and "gaze copy" in doc["node.efference"]["w"]
    tr = inspect.getsource(PS.transition)
    assert "_gc = gaze_copy_of(prediction_heads)" in tr
    assert "out_tokens = _gc(out_tokens, last_step, executed_action, achieved_state)" in tr
    assert '"gaze_copy_into_vision": model.efference.vision_idx is not None' in inspect.getsource(EF.stamp)
    assert "physical_cost = PhysicalGoalCost(self.efference, physical_goal['pose']," in \
        inspect.getsource(LM.LemLite._plan_chain)
    lsrc = open(inspect.getfile(LM), encoding="utf-8").read()
    writes = re.findall(r"\[['\"]physical_goal['\"]\]\s*=", lsrc)
    assert len(writes) == 1 and re.search(
        r"if self\.physical_goals == 'memory':\s*\n\s*self\._last_plan_inputs\['decision'\]"
        r"\['physical_goal'\] = physical_memory", lsrc), \
        "a decision's physical goal is stored somewhere other than under --physical-goals memory"
    for pg, needle in ((None, "do not record --physical-goals"),
                       ("off", "action search never calls the body head"),
                       ("memory", "scores each candidate command")):
        run = _fixture_run_gaze_copy(tmp_path / str(pg), pg)
        if pg == "memory":
            # the search runs only with the planner on (efference_routes; the trainer's --intention
            # default is off): the review of 2026-09-25, finding 3
            _set_args(run, intention="on")
        b = _blocks(BX.render(run))
        page_doc, meta = b["data-doc"], b["data-meta"]
        assert meta["args"]["physical_goals"] == pg
        assert needle in page_doc["node.efference"]["when"], page_doc["node.efference"]["when"]
        assert "adds the gaze copy's output to the vision prediction" in page_doc["node.efference"]["h"]
        for key in ("node.efference", "edge.efference.heads"):
            assert not re.search(r"\{[a-z][a-z0-9_]*\}", json.dumps(page_doc[key])), page_doc[key]
        slide = next(st for st in b["data-tour"] if st["id"] == "efference")
        assert ["efference", "heads"] in slide["edges"] and "gaze copy" in slide["body"]
        args = json.load(open(os.path.join(run, "args.json")))
        assert ("efference", "heads") in _drawable_edges(meta["receipt"], args)


def test_the_gaze_copy_line_and_the_efference_rows_run_in_a_headless_browser(tmp_path):
    """The same page executed: the gaze copy's line is drawn, the efference box's live rows and
    every popup render, and the page's DOC is the embedded map, entry for entry."""
    CK, exe = _browser_or_skip()
    run = _fixture_run_gaze_copy(tmp_path, "off")
    line_check = """<script>addEventListener('load',()=>{
      if(!document.querySelector('.dg-edge[data-a="efference"][data-b="heads"]'))
        throw Error('The gaze copy line is not drawn');
    });</script>"""
    page = tmp_path / "gaze_copy.html"
    page.write_text(BX.render(run).replace("</body>", line_check + "</body>"),
                    encoding="utf-8", newline="")
    rep = CK.check_page(str(page), exe)
    assert rep["ran"], rep.get("why")
    assert not rep["problems"], "\n".join(rep["problems"])
    assert rep["doc_changed"] == [] and rep["popups_opened"] == rep["doc_entries"] == len(_page_doc())
    assert rep["boxes_clicked"] == rep["diagram_boxes"]


# the efference heads' two routes into the planner (2026-09-25): the settings of production's
# segments (H84 seg40: --controllability on, --learning-progress on, --wellness-rank normalized,
# --physical-goals off) and the same with --physical-goals memory
_EFF_PLANNER_ARGS = {"controllability": "on", "learning_progress": "on",
                     "wellness_rank": "normalized", "intention": "on"}


def test_efference_routes_follow_the_code_and_the_launch():
    """schema.efference_routes decides the two lines out of the efference box into the planner, the
    tour step that animates them and the popups' run sentences. Its rule is held here to the code
    that makes each path, and read over a table of settings: the goal ranking reads the body head's
    order-long expectation (controllability) and both heads' errors against doing nothing (learning
    progress) only under --wellness-rank normalized, and the action search calls the body head only
    under --physical-goals memory with the planner on."""
    import inspect
    from lem.model import efference as EF
    from lem.model import wellness_channels as WC
    from mimo_bridge import lem_model as LM
    SCH = BX.SCH
    assert SCH.EFFERENCE_RANK_CHANNELS == {c: WC.DEFAULT_WEIGHTS[c] for c in SCH.EFFERENCE_RANK_CHANNELS}
    # the paths in the code: the tracker is handed the body head's tenure expectation, both heads'
    # errors are learning-progress keys, and the channel vector is built by the normalised ranking only
    obs = inspect.getsource(LM.LemLite._observe_wellness_channels)
    assert "self.last_efference_tenure)" in obs and "(self._eff_lp or {}).items()" in obs
    les = inspect.getsource(EF._efference_lesson)
    assert 'model._eff_lp["efference_body"]' in les and 'model._eff_lp["efference_gaze"]' in les
    lsrc = open(inspect.getfile(LM), encoding="utf-8").read()
    assert lsrc.count("self._wellness_channel_vector(") == 1, "the channel vector gained a second caller"
    assert "raw = self._wellness_channel_vector(" in inspect.getsource(LM.LemLite._rank_store_moment)
    assert lsrc.count("self._rank_store_moment(") == 1 and re.search(
        r'if self\.wellness_rank_mode == "normalized":[^\n]*\n(?:\s*#[^\n]*\n)*\s*_chv = self\._rank_store_moment\(',
        lsrc), "the channels reach the goal ring outside the normalised ranking"
    assert 'w["controllability"] = 0.0' in inspect.getsource(LM.LemLite._rank_weights_now)
    # the table
    eff = {"efference_heads": {"weight": 0.1}}
    R = SCH.efference_routes
    assert R({}, dict(_EFF_PLANNER_ARGS, physical_goals="memory")) == \
        {"goal_channels": [], "action_search": False}, "no heads, no route"
    assert R(eff, dict(_EFF_PLANNER_ARGS, physical_goals="off")) == \
        {"goal_channels": ["controllability", "learning_progress"], "action_search": False}
    assert R(eff, dict(_EFF_PLANNER_ARGS, wellness_rank="raw"))["goal_channels"] == []
    assert R(eff, dict(_EFF_PLANNER_ARGS, learning_progress="off"))["goal_channels"] == ["controllability"]
    assert R(eff, dict(_EFF_PLANNER_ARGS, wellness_weights="controllability=0,learning_progress=0"))[
        "goal_channels"] == []
    # the model's own stamp wins over the launch flags
    stamp = dict(eff, wellness_channels={"rank": "normalized", "channels": ["integrity", "learning_progress"],
                                         "weights": {"controllability": 1.0, "learning_progress": 0.75}})
    assert R(stamp, {"wellness_rank": "raw", "controllability": "on"})["goal_channels"] == ["learning_progress"]
    assert R(eff, {"physical_goals": "memory", "intention": "on"})["action_search"] is True
    assert R(eff, {"physical_goals": "memory", "intention": "off"})["action_search"] is False
    assert R(eff, {"physical_goals": "off", "intention": "on"})["action_search"] is False


def test_the_efference_lines_into_the_planner_follow_the_run(tmp_path):
    """Production's settings draw the dashed line into the wellness box (the goal ranking) and no
    action-search line; --physical-goals memory adds the solid line into the planner; a run that
    ranks by the raw sum draws neither. The tour step exists exactly when a line does and animates
    only drawn lines, and each line's popup names the run's own settings."""
    for name, extra, want in (
            ("production", {"physical_goals": "off"}, {("efference", "wellness")}),
            ("search", {"physical_goals": "memory"}, {("efference", "wellness"), ("efference", "intention")}),
            ("raw", {"physical_goals": "off", "wellness_rank": "raw"}, set())):
        run = _fixture_run_gaze_copy(tmp_path / name, extra["physical_goals"])
        args = _set_args(run, **dict(_EFF_PLANNER_ARGS, **extra))
        b = _blocks(BX.render(run))
        meta, doc, tour = b["data-meta"], b["data-doc"], b["data-tour"]
        drawable = _drawable_edges(meta["receipt"], args)
        assert {e for e in drawable if e[0] == "efference" and e[1] in ("wellness", "intention")} == want, name
        assert meta["efference_routes"] == BX.SCH.efference_routes(meta["receipt"], args)
        step = next((st for st in tour if st["id"] == "efference_planner"), None)
        assert (step is not None) == bool(want), name
        for st in tour:
            for a, to in st["edges"]:
                assert (a, to) in drawable, f"{name}: slide {st['id']} animates an undrawn line {a}->{to}"
        if step is not None:
            assert {tuple(e) for e in step["edges"] if e[0] == "efference"} == want
        for key in ("edge.efference.wellness", "edge.efference.intention", "node.efference"):
            assert not re.search(r"\{[a-z][a-z0-9_]*\}", json.dumps(doc[key])), (name, key)
        chan = doc["edge.efference.wellness"]["w"]
        if ("efference", "wellness") in want:
            assert "the goal ranking reads controllability and learning progress" in chan
        else:
            assert "ranks goals by the raw sum (--wellness-rank raw)" in chan
        search = doc["edge.efference.intention"]["w"]
        assert ("was launched with --physical-goals memory" in search) == (("efference", "intention") in want)


def test_the_efference_lines_into_the_planner_run_in_a_headless_browser(tmp_path):
    """The three pages executed: each draws exactly the lines its routes name, the seventh corridor
    only with the action-search line, and every popup, box and slide renders clean."""
    CK, exe = _browser_or_skip()
    for name, pg, rank in (("production", "off", "normalized"), ("search", "memory", "normalized"),
                           ("raw", "off", "raw")):
        run = _fixture_run_gaze_copy(tmp_path / name, pg)
        _set_args(run, **dict(_EFF_PLANNER_ARGS, physical_goals=pg, wellness_rank=rank))
        want_w, want_i = rank == "normalized", pg == "memory"
        check = ("""<script>addEventListener('load',()=>{
          const on=(a,b)=>!!document.querySelector('.dg-edge[data-a="'+a+'"][data-b="'+b+'"]');
          if(on('efference','wellness')!==WANT_W) throw Error('efference->wellness contradicts the routes');
          if(on('efference','intention')!==WANT_I) throw Error('efference->intention contradicts the routes');
          const e=document.querySelector('.dg-edge[data-a="efference"][data-b="wellness"]');
          if(e && !e.classList.contains('loopback')) throw Error('the goal-ranking line must be dashed');
          const s=document.querySelector('.dg-edge[data-a="efference"][data-b="intention"]');
          if(s && s.classList.contains('loopback')) throw Error('the action-search line must be solid');
          // the box's live row (review of 2026-09-25, finding 1): the action search is named apart
          // from the goal route, and a page that draws the dashed line says the heads reach the command
          const row=(liveValuesFor('efference')||[]).find(r=>r[0]==="body head in the planner's action search");
          if(!row) throw Error('the efference box has no action-search row');
          if(/does not by itself change/.test(row[1])) throw Error('the removed clause is back: '+row[1]);
          if(/through the goal ranking/.test(row[1])!==(WANT_W && !WANT_I))
            throw Error('the goal-route clause contradicts the routes: '+row[1]);
          // the goal-ranking line crosses two older dashed lines, so it is drawn over them on a white
          // halo (review of 2026-09-25, finding 12): the halo exists with the line, follows every
          // line it crosses, and the line follows the halo
          const halo=document.querySelector('.dg-halo[data-for="efference-wellness"]');
          if(!!halo!==WANT_W) throw Error('the halo of the goal-ranking line contradicts the routes');
          if(halo){
            for(const [a,b] of [['motor','imagination'],['curriculum','world'],['motor','contact_head']]){
              const x=document.querySelector('.dg-edge[data-a="'+a+'"][data-b="'+b+'"]');
              if(x && !(x.compareDocumentPosition(halo)&Node.DOCUMENT_POSITION_FOLLOWING))
                throw Error('the line '+a+'->'+b+' is drawn over the halo');
            }
            if(!(halo.compareDocumentPosition(e)&Node.DOCUMENT_POSITION_FOLLOWING))
              throw Error('the goal-ranking line is drawn under its own halo');
          }
        });</script>""".replace("WANT_W", json.dumps(want_w)).replace("WANT_I", json.dumps(want_i)))
        page = tmp_path / (name + ".html")
        page.write_text(BX.render(run).replace("</body>", check + "</body>"), encoding="utf-8", newline="")
        rep = CK.check_page(str(page), exe)
        assert rep["ran"], (name, rep.get("why"))
        assert not rep["problems"], name + ":\n" + "\n".join(rep["problems"])
        assert rep["doc_changed"] == [] and rep["popups_opened"] == rep["doc_entries"]
        assert rep["tour_rendered"] == rep["tour_slides"]


def test_the_physical_goals_sentence_names_the_goal_route_beside_the_action_search(tmp_path):
    """Review of 2026-09-25, findings 1 and 3. On production's settings (--physical-goals off, goals
    ranked normalized by controllability and learning progress) the page said a more accurate body
    prediction does not by itself change which order is chosen, beside the dashed line that shows
    the heads ranking the goal: a different goal is a different order, and learning progress has no
    evidence gate. The sentence now keeps only what holds (the action search never calls the body
    head) and names the goal route wherever the routes draw it. --physical-goals memory with the
    planner off stores no physical goal (step() stores it inside the planning block), so its search
    sentence is not printed there."""
    removed = "does not by itself change which command is chosen"
    assert removed not in _viewer_js()
    for name, extra, route in (
            ("production", {"physical_goals": "off"}, True),
            ("search", {"physical_goals": "memory"}, True),
            ("raw", {"physical_goals": "off", "wellness_rank": "raw"}, False),
            ("planner_off", {"physical_goals": "memory", "intention": "off", "wellness_rank": "raw"}, False)):
        run = _fixture_run_gaze_copy(tmp_path / name, extra["physical_goals"])
        args = _set_args(run, **dict(_EFF_PLANNER_ARGS, **extra))
        b = _blocks(BX.render(run))
        doc, meta = b["data-doc"], b["data-meta"]
        when = doc["node.efference"]["when"]
        assert removed not in json.dumps(doc), name
        assert meta["efference_routes"] == BX.SCH.efference_routes(meta["receipt"], args)
        assert bool(meta["efference_routes"]["goal_channels"]) is route, name
        assert ("reach the choice of command through the goal" in when) is route, (name, when)
        searched = meta["efference_routes"]["action_search"]
        assert searched is (name == "search"), name
        assert ("scores each candidate command" in when) is searched, (name, when)
        assert ("action search never calls the body head" in when) is (name in ("production", "raw")), (name, when)
        if name == "planner_off":
            assert "its planner is off (--intention off)" in when and "no action search runs" in when
        slide = next((st for st in b["data-tour"] if st["id"] == "efference_planner"), None)
        if name == "search":
            assert "the planner also calls the body head" in slide["body"]


def test_the_controllability_and_gaze_copy_popups_follow_the_code():
    """Review of 2026-09-25, findings 4 and 5. Controllability is a running average with the newest
    measured order at one fifth (lem/model/wellness_channels.py ControllabilityTracker, alpha
    1/CONTROL_EMA_ORDERS, divided by its accumulated weight), not an average of the last five: once
    many orders are measured the last five carry 1 - 0.8^5 of it. And the gaze copy is trained by
    re-study and the imagination chain only where they run (--replay; --imagination)."""
    import inspect
    from lem.model import wellness_channels as WC
    doc = _page_doc()
    w = doc["edge.efference.wellness"]["w"]
    assert WC.CONTROL_EMA_ORDERS == 5 and "counts one fifth" in w
    assert f"the last five carry {round(100 * (1 - (1 - 1 / WC.CONTROL_EMA_ORDERS) ** 5))} of every 100 parts" in w
    assert "averaged over the body's last 5" not in json.dumps(doc)
    src = inspect.getsource(WC.ControllabilityTracker)
    assert "self.alpha = 1.0 / float(max(1, int(ema_orders)))" in src
    assert "(1 - a) * self.ema_m + a * agr" in src and "self.ema_m / self.ema_w.clamp_min(1e-12)" in src
    # the misreading line names the settings under which the heads reach the goal (the reviewer's
    # second pass over finding 1)
    assert ("Under --wellness-rank normalized with --controllability or --learning-progress on, the "
            "heads' readings change which remembered model step becomes the goal") in " ".join(
        doc["node.efference"]["f"].split())
    when = doc["node.efference"]["when"]
    assert "so all three train it" not in when
    assert "where they run, before re-study (--replay on" in when and "imagination chain (--imagination)" in when


def test_motor_facts_come_from_the_receipt(tmp_path):
    run = _fixture_run(tmp_path)
    b = _blocks(BX.render(run))
    motor = next(st for st in b["data-tour"] if st["id"] == "motor")
    assert "26-handle" in motor["body"]
    assert "Handle 23 of the 26" in motor["body"]
    assert "24-handle" not in motor["body"]


def test_voice_slide_says_no_lesson_reaches_the_head(tmp_path):
    """The fixture receipt enables the voice head, so the (final_norm, voice) line
    and the voice slide are actually exercised - and the slide must not present
    weights that no lesson reaches as a working voice."""
    run = _fixture_run(tmp_path)
    b = _blocks(BX.render(run))
    voice = next(st for st in b["data-tour"] if st["id"] == "voice")
    assert "No lesson reaches it" in voice["body"], \
        "the slide must say no lesson reaches the head"
    assert "93,264" in voice["body"], "the count comes from the receipt, not the text"
    assert "make no sound" in voice["body"], \
        "the slide must say the controls drive nothing in the MIMo world yet"
    assert "vocal-tract controls" not in voice["body"] and "controls of a simulated vocal tract" not in voice["body"]
    assert "56" in voice["body"], "the slide must name the width the repo's vocal tract actually has"
    assert ["final_norm", "voice"] in [list(e) for e in voice["edges"]], \
        "the slide must animate the line the diagram draws to the voice box"


def _page_doc():
    """The doc map the page actually ships: schema.DOC merged with the chart
    entries, exactly as render() builds it."""
    doc = dict(BX.SCH.DOC)
    doc.update(BX.DOC_CHARTS)
    return doc


def _page_sources():
    """Every shipped source that can hand a doc key to the page's popup opener."""
    import inspect
    from mimo_bridge.explainer import walkthrough as WT
    return (open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
            + inspect.getsource(WT) + inspect.getsource(BX))


def test_every_doc_marker_resolves_and_no_glossary_entry_is_unreachable():
    """The page's factual layer, held by the suite. Every [[marker]] must open an
    entry that exists, and every glossary entry must be reachable from some doc
    text or from a key the drawing code passes - an entry nothing links to is a
    popup no reader can open, and its wording drifts from whatever text replaced
    it."""
    doc = _page_doc()
    referenced = set()
    for entry in doc.values():
        for field in entry.values():
            for marker in re.findall(r"\[\[([^\]]+)\]\]", str(field)):
                referenced.add("term." + marker.split("|")[0])
    assert sorted(k for k in referenced if k not in doc) == [], \
        "a [[marker]] opens a glossary entry that does not exist"
    src = _page_sources()
    unreachable = sorted(k for k in doc if k.startswith("term.")
                         and k not in referenced and '"%s"' % k not in src)
    assert unreachable == [], "glossary entries nothing on the page links to"


def _doc_key_for(node_id):
    """Mirror of viewer.js docKeyFor: the entry a drawn box opens."""
    m = re.match(r"^block\d+_(s\d|ffn)$", node_id)
    if m:
        return "node.block_" + m.group(1)
    if re.match(r"^(sense|enc)_", node_id):
        return "node." + node_id.replace("_", ".", 1)
    return "node." + node_id


def _edge_doc_key(a, b):
    """Mirror of viewer.js edgeKey: the entry a drawn line opens."""
    x = re.sub(r"^block\d+_", "block_", a)
    y = re.sub(r"^block\d+_", "block_", b)
    if x.startswith("enc_") and y == "ln_out":
        x = "enc"
    if x.startswith("sense_") and y.startswith("enc_"):
        y = "enc"
    return f"edge.{x}.{y}"


# Every box buildDiagram can draw. The ones it builds from a literal id are checked
# against viewer.js below; the strip rows come from arrays and are carried by hand,
# the same way _drawable_edges carries the per-sense and per-block lines.
_DRAWABLE_BOXES = ["world", "ln_out", "td_gate", "assembly", "final_norm", "heads",
                   "recon", "motor", "gaze", "wellness",
                   "curiosity", "intention", "imagination", "replay", "curriculum",
                   "memory", "self_state", "voice", "hself", "caregiver",
                   # the gain-of-function build (2026-09-22)
                   "entity_self", "efference", "contact_head", "synergy", "synergy_examples"]


def test_every_doc_entry_can_be_opened_by_some_click():
    """An entry the page ships that no click opens is a popup no reader can reach,
    and its wording drifts from whatever text replaced it. The term.* half of that
    is held above; this holds the other 120-odd entries - the boxes, the lines, the
    charts and the drawn captions - by enumerating every way the page opens a popup:
    a key written literally in the shipped sources, a drawn box through docKeyFor, a
    drawn line through edgeKey, and the [[term]] links inside any text those reach.
    Five entries were unreachable on 2026-09-15, two of them the data contracts of
    the two lines around the top-down gate."""
    doc = _page_doc()
    src = _page_sources()
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    diagram = js[js.index("function buildDiagram()"):js.index("function buildLearnLoop()")]
    # only calls whose id is one whole literal; node("sense_" + m, ...) builds its id
    # from an expression and the per-sense and per-block boxes are carried by hand
    literal_boxes = set(re.findall(r'\bnode\(\s*"([A-Za-z0-9_]+)"\s*,', diagram))
    assert literal_boxes <= set(_DRAWABLE_BOXES), \
        f"buildDiagram draws boxes this list never names: {sorted(literal_boxes - set(_DRAWABLE_BOXES))}"
    mods = ["vision", "tactile", "audio", "proprioception", "motor", "intermind"]
    # the sixth slot is intermind or, under --vestibular-sense on, vestibular: both configurations
    # draw their own sense box, encoder box and lines
    mods_v = mods[:5] + ["vestibular"]
    boxes = (_DRAWABLE_BOXES
             + [f"sense_{m}" for m in mods + ["vestibular"]]
             + [f"enc_{m}" for m in mods + ["vestibular"]]
             + [f"block{i}_{s}" for i in (0, 1) for s in ("s1", "s2", "s3", "ffn")])
    edges = set()
    for sixth in (mods, mods_v):
        receipt = {"active_modalities": sixth, "forward_model": {"n_layers": 2},
                   "persistent_memory": "on", "voice_head": {"enabled": True},
                   "efference_heads": {"weight": 0.1, "gaze_copy_into_vision": True},
                   "contact_head": {"weight": 0.1},
                   "motor_synergy": {"dim": 16, "babble": True}}
        for gate in ("off", "on"):          # the two gate settings draw different lines
            edges |= _drawable_edges(receipt, {"hierarchical_self": "dark",
                                               "top_down_gate": gate, "entity_self": "dark",
                                               "wellness_rank": "normalized",
                                               "controllability": "on",
                                               "physical_goals": "memory", "intention": "on",
                                               "caregiver_mode": "on", "self_state": "learn",
                                               "residual_prediction": "on", "consequence_w": 0.5})
    reach = {k for k in doc if '"%s"' % k in src}
    reach |= {_doc_key_for(n) for n in boxes}
    reach |= {_edge_doc_key(a, b) for a, b in edges}
    grew = True
    while grew:                          # [[term]] links open their own popup
        grew = False
        for key in list(reach):
            for field in doc.get(key, {}).values():
                for marker in re.findall(r"\[\[([^\]|]+)", str(field)):
                    term = "term." + marker
                    if term in doc and term not in reach:
                        reach.add(term)
                        grew = True
    missing = sorted(k for k in doc if k not in reach)
    assert not missing, f"entries the page ships that no click can open: {missing}"


def test_glossary_numbers_come_from_the_code_that_owns_them():
    """Numbers the glossary quotes are recomputed here from the constants that
    produce them, so a constant cannot move without this failing."""
    import inspect
    from lem.skin_mechanics import PROFILE
    from lem.model.wellness import WellnessEvaluators
    from mimo_bridge.movement_drive import MovementDrive
    doc = _page_doc()

    # The linked mechanism gives the exact reference and exposure onset, not rounded old prose.
    touch = doc["term.mechanical-load"]
    assert f"{PROFILE.pressure_reference_pa / 1000:g} kPa" in touch["w"]
    assert PROFILE.pressure_exposure_start_fraction == 0.25 and "a quarter" in touch["f"]
    assert f"{PROFILE.pressure_exposure_seconds:g} seconds" in touch["w"]
    assert "L/(1+L)" in touch["w"]

    # posture: the reach in centimetres and the boundary zone are BODY_REACH and the
    # 10% threshold, read out of the formula that uses them
    wsrc = inspect.getsource(WellnessEvaluators)
    reach = float(re.search(r"BODY_REACH\s*=\s*([\d.]+)", wsrc).group(1))
    pos_range_cm = float(re.search(r"(\d+)\s*cm position range", wsrc).group(1))
    zone = float(re.search(r"threshold\s*=\s*([\d.]+)\s*\*\s*rom", wsrc).group(1))
    posture = doc["term.eval-motor-state"]["w"]
    reach_cm = round(reach * pos_range_cm, 1)
    assert f"{reach_cm:g} cm" in posture, f"the entry no longer quotes {reach_cm:g} cm of reach"
    assert f"final {round(zone * 100):g}%" in posture, "the boundary penalty band moved"
    assert f"first {round((1 - zone) * 100):g}%" in posture
    assert "not proposed target positions or joint-limit angles" in posture

    # movement: the two times the entry gives are the drive's own defaults
    sig = inspect.signature(MovementDrive.__init__).parameters
    move = doc["term.eval-movement"]["w"]
    assert f"{sig['debt_rise_s'].default:g} seconds" in move
    assert f"{sig['debt_fall_s'].default:g} seconds" in move
    assert sig["still_thresh"].default == 0.25 and "a quarter of the reference speed" in move

    # the collapse watch: window, stride and the body time they add up to
    from mimo_bridge import lem_model as LM
    lsrc = open(inspect.getfile(LM), encoding="utf-8").read()
    ring = int(re.search(r"self\.token_ring[^\n]*maxlen=(\d+)", lsrc).group(1))
    stride = int(re.search(r"self\.step_count % (\d+) == 0:\n\s+for m in self\.active:"
                           r"\n\s+self\.token_ring", lsrc).group(1))
    collapse = doc["term.collapse"]["h"]
    assert f"last {ring} stored samples" in collapse
    assert f"every {stride}th" in collapse
    assert f"{ring * stride:,} model steps" in collapse
    # the window in SECONDS is this run's own since the H84 items (2026-09-19): the entry
    # carries a placeholder, run_fill multiplies the code's window by the run's measured
    # model step, and the constant it multiplies is held to the code here
    assert BX.SCH.COLLAPSE_WINDOW_STEPS == ring * stride
    assert "{collapse_window_s} seconds of body time" in collapse
    assert "50 seconds" not in collapse and "50 seconds" not in doc["term.collapse"]["f"]
    assert BX.SCH.run_fill({}, {}, "R")["{collapse_window_s}"] == f"{ring * stride * 0.05:g}"
    assert BX.SCH.run_fill({"env_dt_s": 0.02, "model_step_s": 0.06}, {"action_repeat": 3},
                           "R")["{collapse_window_s}"] == f"{ring * stride * 0.06:g}"


def test_the_learning_loop_text_matches_the_code():
    """The learning-loop circle is the page's account of how the model learns and
    nothing checked it against the code: three claims that are guarded on the CODE
    side - two optimizer groups, the grad-enabled replay motor re-encode, no replay
    VICReg - could drift in the prose with nothing to catch it."""
    import inspect
    from mimo_bridge import lem_model as LM
    doc = _page_doc()

    # 1. re-study DOES train the motor encoder: the step-from motor row is
    #    re-encoded with gradient enabled, while the context rebuild is not.
    ru = inspect.getsource(LM.LemLite.replay_update)
    seg = ru[ru.index("def _enc"):ru.index('cur["motor"] = torch.where(')]
    assert '{"motor": flat}, {"motor": "real"}' in seg, \
        "the re-encode no longer goes through the registry"
    assert "no_grad" not in seg, "the step-from motor re-encode must keep its gradient"
    assert "with torch.no_grad():" in ru and "reb = _enc(" in ru, \
        "the context rebuild must stay detached"

    # 2. no VICReg term in re-study - removed 2026-09-06 because it graded stored
    #    fp16 ring contents that reach no parameter
    assert "compute_vicreg" not in ru, "replay has run no VICReg term since 2026-09-06"
    assert "remembered tokens are detached" in doc["bp.replay"]["w"]

    # 3. the page may not say re-study reaches NO encoder: five of the six are
    #    excluded and the motor one is trained. Whole-line // comments are dropped
    #    first - one of them quotes the old wrong wording to record that it was fixed.
    prose = re.sub(r"^[ \t]*//.*$", "", _page_sources(), flags=re.M)
    for sentence in re.findall(r"[^.]*encoders?[^.]*excluded[^.]*", prose):
        assert re.search(r"\bfive\b|\b5\b|not excluded", sentence, re.I), \
            f"the page says re-study reaches no encoder at all: {sentence.strip()[:120]}"

    # 4. The shortened entry describes the gradient route. Parameter counts remain
    #    receipt-derived, independently of whether this popup prints them.
    replay_txt = doc["bp.replay"]["w"]
    assert "current motor encoder" in replay_txt
    assert "other sense encoders do not receive this replay gradient" in replay_txt
    shape = BX.SCH.shape_fill({"total_params": 5728601,
                              "component_params": {"enc_motor": 285184}}, {})
    for key in ("{enc_motor_params}", "{enc_motor_pct}", "{total_params}"):
        assert key in doc["node.replay"]["w"]
    trained = int(shape["{enc_motor_params}"].replace(",", ""))
    total = int(shape["{total_params}"].replace(",", ""))
    share = float(shape["{enc_motor_pct}"].rstrip("%"))
    assert abs(100 * trained / total - share) < 0.05

    # 5. Keep the optimizer-group and configured-default regression checks; the
    #    overview no longer promises those defaults for every selected run.
    init = inspect.getsource(LM.LemLite.__init__)
    assert 'p for p in params if p.dim() >= 2' in init and \
           'p for p in params if p.dim() < 2' in init, \
        "the optimizer is no longer split into two groups by tensor shape"
    assert '{"params": no_decay, "weight_decay": 0.0}' in init, \
        "the 1-D group no longer carries weight decay 0.0"
    import copy
    import lem
    import yaml
    from lem.config.resolver import resolve
    tr = resolve(copy.deepcopy(yaml.safe_load(
        open(os.path.join(os.path.dirname(inspect.getfile(lem)), "master_config.yaml"),
             encoding="utf-8"))))["training"]
    update = doc["bp.update"]["w"]
    assert (tr["lr"], tr["weight_decay"], list(tr["betas"])) == (3e-4, 0.01, [0.9, 0.999])
    assert "saved optimizer state" in update and "Evaluation-only runs skip" in update
    assert "Planning happens afterward" in update


def test_the_learning_loop_circle_reads_the_run_and_its_arcs_clear_the_boxes(tmp_path):
    """The circle's numbers and its frozen-or-live state must come from the run's
    own args and snapshots, and its arcs must land outside the station boxes.
    JavaScript is not executed here, so the check is on the source the page ships
    - the same hand-written mirror idiom as _drawable_edges above."""
    import math
    run = _fixture_run(tmp_path)           # 3 snapshots, all stamped weights "frozen"
    _set_args(run, intention="on", no_intention_execute=False, hand_goals="memory",
              prospective_wellness="add", prospective_wellness_w=0.15,
              causal_chain="on", wellness_rank="normalized")
    page = BX.render(run)
    goal_js = page[page.index("function buildLearnLoop()"):page.index("function buildWeightLoop(root)")]
    assert 'a.prospective_wellness === "add" && Number(a.prospective_wellness_w) > 0' in goal_js
    assert 'a.causal_chain === "on"' in goal_js
    assert 'a.wellness_rank === "normalized"' in goal_js
    assert 'buildWeightLoop(details)' in goal_js
    data = _blocks(page)
    for key in ("loop.overview", "loop.goal", "loop.search", "loop.execute",
                "loop.outcome", "loop.memory", "loop.learning"):
        assert key in goal_js and key in data["data-doc"]
    for arg, value in (("no_intention_execute", False), ("hand_goals", "memory"),
                       ("prospective_wellness", "add"), ("prospective_wellness_w", 0.15),
                       ("causal_chain", "on"), ("wellness_rank", "normalized")):
        assert data["data-meta"]["args"][arg] == value
    js = page[page.index("function buildWeightLoop(root)"):]
    js = js[:js.index("\n}\n")]
    for arg in ("action_repeat", "replay_every", "replay_batch", "eval_only", "n_envs"):
        assert arg in js, f"the circle never reads the run's {arg}"
    for literal in ("~50 ms", "every 4th model step", "48 remembered windows",
                    "slightly better weights"):
        assert literal not in js, f"the circle still hardcodes {literal!r}"
    # the fixture's snapshots all read "frozen", so the drawing must be able to
    # show a run in which no weight moved instead of asserting improvement
    assert "did NOT run: the weights are frozen" in js, \
        "the circle cannot say stations 3 and 4 were skipped on an --eval-only run"
    assert "did not run and no weight changed" in js
    assert "no weight updates from replay" in js, \
        "re-study's box cannot say the ring never moved a weight"
    # geometry mirror: an arc endpoint under a station box is an arrowhead the box -
    # drawn after the arcs - paints out. The angles and the box widths are READ OUT
    # of the drawing code rather than pinned here, so a re-trimmed arc is checked at
    # its new angle instead of failing for having moved at all.
    cx, cy, rx, ry = 430, 132, 310, 88
    assert f"const cx = {cx}, cy = band + {cy}, rx = {rx}, ry = {ry};" in js
    env = {"cx": cx, "cy": cy, "rx": rx, "ry": ry}

    def _val(expr):                      # "cx", "cy - ry", "cx + rx" and nothing else
        m = re.fullmatch(r"(\w+)(?:\s*([-+])\s*(\w+))?", expr.strip())
        assert m, f"a station is placed at an expression this mirror cannot read: {expr}"
        v = env[m.group(1)]
        return v if not m.group(2) else (v + env[m.group(3)] if m.group(2) == "+"
                                         else v - env[m.group(3)])

    boxes = [(_val(x), _val(y), int(w)) for x, y, w in
             re.findall(r"station\(([^,]+), ([^,]+), (\d+),", js)]
    assert len(boxes) == 4, f"{len(boxes)} station boxes parsed, 4 expected"
    arcs = [(float(a), float(b)) for a, b in
            re.findall(r'^\s*\[(-?[\d.]+), (-?[\d.]+), "edge\.bp\.', js, re.M)]
    assert len(arcs) == 4, f"{len(arcs)} arcs parsed, 4 expected"
    for pair in arcs:
        for ang in pair:
            x = cx + rx * math.cos(math.radians(ang))
            y = cy + ry * math.sin(math.radians(ang))
            for bx, by, bw in boxes:
                assert not (bx - bw / 2 <= x <= bx + bw / 2 and by - 21 <= y <= by + 21), \
                    f"the arc endpoint at {ang} degrees lands under a station box"
    # every popup the circle can open exists in the page's doc block
    doc = _blocks(page)["data-doc"]
    opened = sorted(set(re.findall(r'"((?:edge\.)?bp\.[a-z_.]+)"', js)))
    assert len(opened) >= 11, f"only {len(opened)} clickable doc keys in the circle"
    for key in opened:
        assert key in doc, f"the circle opens {key}, which has no doc entry"


def test_the_headline_chart_does_not_squash_rank1_under_the_decoy_count():
    """P443: one linear axis carried a 0-1 score and a constant count of 14.

    rank1 is the number this project exists to produce and it lives in 0 to 1.
    Plotting the raw decoy count beside it ran the axis to 14, leaving the
    headline line in the bottom 7% of the plot. What the reader needs from that
    count is the line rank1 must beat, 1/(1+decoys), which shares rank1's scale.
    """
    recs = [{"step": i, "tenure_tok": {"rank1": 0.30 + 0.001 * i,
                                       "n_decoys": 14, "vis_gate_open": 0.4}}
            for i in range(40)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    headline = [c for c in out["charts"] if "headline" in c["label"]]
    assert headline, "the headline chart did not build"
    series = headline[0]["series"]
    labels = [s["label"] for s in series]
    assert any("guessing" in l for l in labels), "no chance line is plotted"
    assert not any("decoy count" == l for l in labels), "the raw count is back on the axis"

    axis_top = max(p[1] for s in series for p in s["points"])
    assert axis_top <= 1.0, f"axis runs to {axis_top}, so something off-scale is plotted"

    chance = [s for s in series if "guessing" in s["label"]][0]
    # _series rounds every plotted value to 6 decimals
    assert all(abs(p[1] - 1 / 15) < 1e-6 for p in chance["points"])
    rank1 = [s for s in series if s["key"] == "tenure_tok.rank1"][0]
    assert max(p[1] for p in rank1["points"]) / axis_top > 0.30


def test_the_chance_line_follows_the_run_s_own_decoy_count():
    """It is read per window, not assumed, so a run that changes it moves the line."""
    recs = [{"step": i, "tenure_tok": {"rank1": 0.3, "n_decoys": n}}
            for i, n in enumerate([14] * 10 + [3] * 10)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    chance = [s for c in out["charts"] if "headline" in c["label"]
              for s in c["series"] if "guessing" in s["label"]][0]
    values = sorted({round(p[1], 6) for p in chance["points"]})
    assert values == [round(1 / 15, 6), round(1 / 4, 6)]


def test_a_missing_decoy_count_drops_the_chance_line_without_breaking_the_chart():
    recs = [{"step": i, "tenure_tok": {"rank1": 0.3}} for i in range(20)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    headline = [c for c in out["charts"] if "headline" in c["label"]][0]
    assert [s["key"] for s in headline["series"]] == ["tenure_tok.rank1"]


def test_the_page_does_not_say_observers_learn_nothing():
    """P403: 'learn nothing' conflated carrying no gradient with not changing.

    The hierarchical self holds no weights at all - it is buffers - and those
    buffers change every time it observes. That IS how it earns its evidence.
    What carrying no gradient means is that it cannot change what the body does,
    which is a different and much narrower claim.
    """
    import inspect
    import re
    from mimo_bridge import hierarchical_self
    from mimo_bridge.explainer.schema import DOC

    source = inspect.getsource(hierarchical_self)
    # the count the page states is the number of buffer INSTANCES a built tracker
    # holds (per-level buffers are registered in a loop), not register_buffer calls
    n_buffers = len(list(hierarchical_self.HierarchicalSelfDark(["a", "b"], 8, n_slots=8).buffers()))
    assert re.search(r"nn\.Parameter", source) is None, "it now has weights"
    assert n_buffers > 0

    text = DOC["bp.replay"]["w"]
    assert "Observers learn nothing" not in text
    assert "their remembered tokens are detached" in text
    assert "stored patterns" in DOC["node.hself"]["w"]
    assert "Nothing uses the count yet" in DOC["node.hself"]["w"]


def test_a_frozen_run_does_not_score_its_own_trend_as_a_circle_improvement():
    """P404: the learning-loop picture said two contradictory things at once.

    On an --eval-only page whose charts are its own probe, the caption four lines
    above states "no gradient was computed, no weight moved" - and the verdict line
    then measured that run's next-moment-error trend and called it IMPROVING, as
    though a circle had done it. A frozen model meeting a changing world can move
    that number without any weight changing, so the trend is real but it is not a
    circle's doing. The page now says which it is.

    String checks on the shipped source, the idiom this file already uses because
    the JavaScript is not executed here.
    """
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    loop = js[js.index("function buildWeightLoop"):]
    loop = loop[:loop.index("\n}\n")]

    # the already-landed half: direction is measured, never asserted
    assert 'winStats(seriesByKey("causal_pe"))' in loop, (
        "the circle asserts its direction instead of measuring it with the page's "
        "own trend rule")
    # the phrase may still appear in a COMMENT recording what it replaced; what must
    # never come back is the phrase in a line the reader sees
    drawn = [l for l in loop.splitlines() if not l.lstrip().startswith("//")]
    assert not any("slightly better" in l for l in drawn), (
        "an unmeasured size word is being drawn on the learning-loop picture again")

    # the half this test adds: a frozen page refuses to attribute its trend to a circle
    assert "no weight moved in this run" in loop, (
        "an --eval-only page whose charts are the probe's own still scores a "
        "frozen-weight trend as a circle's improvement")
    assert "not a weight update" in loop
    assert "const probeCharts = frozen &&" in loop, (
        "the frozen branch is not gated on this page's own stats source, so a page "
        "charting a DIFFERENT run that did train would lose its real verdict")


# ---------------------------------------------------------------- audit P112 (2026-09-18)
def test_the_effort_charge_names_this_run_s_body_count(tmp_path):
    """The DOC prose used to hard-code "both bodies" from the two-body probe while the
    training runs have 8. Every {n_bodies} is substituted with the run's own count, and
    neither the literal "both bodies" nor an unsubstituted placeholder can ship."""
    run = _fixture_run(tmp_path)
    args_path = os.path.join(run, "args.json")

    def doc_of(page):
        b = _blocks(page)
        key = next(k for k in b if "doc" in k)
        return b[key]

    doc2 = doc_of(BX.render(run))
    assert "2 bodies" in doc2["term.intention-distance"]["w"]
    assert "2 bodies" in doc2["term.predicted-outcome"]["w"]
    args = json.load(open(args_path))
    args["n_envs"] = 8
    json.dump(args, open(args_path, "w"))
    doc8 = doc_of(BX.render(run))
    assert "8 bodies" in doc8["term.intention-distance"]["w"]
    assert "8 bodies" in doc8["term.predicted-outcome"]["w"]
    for doc in (doc2, doc8):
        for k, v in doc.items():
            for kk, vv in v.items():
                if isinstance(vv, str):
                    assert "both bodies" not in vv, (k, kk)
                    assert "{n_bodies}" not in vv, (k, kk)



# ---------------------------------------------------------------- audit P049 (2026-09-18)
def test_the_page_does_not_say_the_wellness_slope_dies_past_the_reach():
    """The straight-through restore keeps a slope past the reach envelope. The page
    entries that said it was flat there, and attributed H81's 11 zero snapshots to
    handles resting comfortably inside, are corrected; this pins both the measurement
    and the prose."""
    import torch
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.explainer.schema import DOC
    m = LemLite(device="cpu", seed=0, imagination=False, motor_state_formula="boundary")
    w = m.wellness
    w.set_developmental_limits(1.0)
    mc = torch.zeros(1, len(m.handle_names), 9)
    mc[..., :3] = 1.5 * 0.21
    mc.requires_grad_(True)
    score = w._formula_motor_state_boundary(mc)
    g = torch.autograd.grad(score.sum(), mc)[0][..., :3]
    assert g.abs().sum() > 0 and (g < 0).all()
    banned = ("returns no gradient at all", "the penalty is flat, so", "the gradient lives only in the zone")
    offenders = [k for k, v in DOC.items() for field in ("w", "when", "h", "f")
                 if any(b in v.get(field, "") for b in banned)]
    assert not offenders, f"the page still says the wellness slope dies past the reach: {offenders}"
    for key in ("node.wellness", "edge.wellness.motor"):
        text = " ".join(DOC[key].get(f, "") for f in ("w", "when", "h", "f"))
        if "of this probe's 20 snapshots" in text or "11 of" in text:
            assert "before the slope past the limit was restored" in text.lower(), key



# ---------------------------------------------------------------- audit P181 (2026-09-18)
def test_the_memory_view_says_what_it_encodes_and_how_far_it_reaches():
    from lem.model.persistent_memory import MemoryHistory
    h = MemoryHistory()                      # lem_model.py builds it with no arguments
    doc = _page_doc()
    text = " ".join(str(v) for v in doc["node.memory"].values())
    assert f"{h.capacity * h.interval_s:g} s" in text
    assert str(h.capacity + 2) in text and str(h.capacity + 1) in text
    src = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    mem = src[src.index('detailFlow("memory"'):src.index('detailFlow("self_state"')]
    assert "MEM_CAP + 2" in mem and "CORE_WIN" in mem
    insp = src[src.index('panel("Memory beyond the window"'):]
    insp = insp[:insp.index('panel("Self perception"')]
    assert "capacity" in insp and "window_seconds" in insp and "encoded_positions" in insp


# ---------------------------------------------------------------- audit P204 (2026-09-18)
def _h83_self_state_tensors():
    """The H83 self_state tensor inventory shapes (d_model 128, action_size 234, 6 senses):
    legacy head = action_encoder 30,208 + transition 99,072 + future 33,670 = 162,950;
    residual = 98,304 + 768 = 99,072; the two calibration buffers = 468."""
    t = {"action_encoder.weight": 128 * 235, "action_encoder.bias": 128,
         "transition.weight_ih": 3 * 128 * 128, "transition.weight_hh": 3 * 128 * 128,
         "transition.bias_ih": 384, "transition.bias_hh": 384,
         "future.0.weight": 128 * 256, "future.0.bias": 128, "future.2.weight": 6 * 128, "future.2.bias": 6,
         "residual.weight": 768 * 128, "residual.bias": 768,
         "site_encoder.0.weight": 128 * 14, "present.weight": 3 * 128}
    out = {k: {"elements": v, "kind": "parameter"} for k, v in t.items()}
    out["grounded.action_center"] = {"elements": 234, "kind": "buffer"}
    out["grounded.action_scale"] = {"elements": 234, "kind": "buffer"}
    return out


def test_idle_params_names_the_self_state_weights_no_lesson_reaches():
    inv = {"checkpoint_inventory": {"components": {"self_state": {"tensors": _h83_self_state_tensors()}}}}
    r = BX.idle_params(dict(inv, self_state="learn", self_prediction="grounded_residual"))
    assert r["self_state"]["elements"] == 262022
    assert sorted(g["elements"] for g in r["self_state"]["groups"]) == [99072, 162950]
    r = BX.idle_params(dict(inv, self_state="perceive", self_prediction="grounded_residual"))
    assert r["self_state"]["elements"] == 162950 and len(r["self_state"]["groups"]) == 1
    r = BX.idle_params(dict(inv, self_state="learn", self_prediction="legacy"))
    assert r["self_state"]["elements"] == 99072 and len(r["self_state"]["groups"]) == 1
    assert BX.idle_params(dict(inv, self_state="perceive", self_prediction="legacy")) == {}
    for combo in (("learn", "grounded_residual"), ("perceive", "grounded_residual"), ("learn", "legacy")):
        r = BX.idle_params(dict(inv, self_state=combo[0], self_prediction=combo[1]))
        assert 468 not in {r["self_state"]["elements"]} | {g["elements"] for g in r["self_state"]["groups"]}



# ---------------------------------------------------------------- audit P248 (2026-09-18)
def test_the_voice_box_text_does_not_call_all_80_outputs_controls():
    w = _page_doc()["node.voice"]["w"]
    assert "80 controls of a simulated vocal tract" not in w
    assert "56" in w and "24" in w


# ---------------------------------------------------------------- audit P255 (2026-09-18)
def test_the_voice_rows_do_not_outprint_bfloat16():
    import math
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    block = js[js.index("if (snap.voice !== undefined)"):js.index("if (snap.caregiver !== undefined)")]
    # DESIGN.md section 11 (2026-09-23): the rows print a percentage of the 0-to-1 range to one
    # decimal (score() - a tenth of a percent, the same resolution the three decimals had)
    for k in ("rms", "min", "max"):
        assert f"score(snap.voice.{k})" in block
    assert "labeledKv(snap.voice," not in block, "the panel went back to labeledKv's raw numbers"
    # bfloat16 keeps 8 significand bits: the spacing below 0.5 is 2^-9, at and above 0.5 it is 2^-8,
    # which the caption writes as percentages of the range, rounded to a tenth
    lo_sp, hi_sp = 2.0 ** -9, 2.0 ** -8
    assert f"{100 * lo_sp:.1f}%" == "0.2%" and f"{100 * hi_sp:.1f}%" == "0.4%"
    assert "0.2% of the range apart below the middle" in block and "0.4% apart at" in block
    assert "0.001953125" in block and "0.00390625" in block, "the step count reads bfloat16's own spacing"


# ---------------------------------------------------------------- audit P414 (2026-09-18)
def test_the_memory_footer_takes_its_model_step_from_the_pages_own_throughput(tmp_path):
    run = _fixture_run(tmp_path)
    page = BX.render(run)
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    assert "564 to 2,346" not in page and "564 to 2,346" not in js
    assert re.search(r"\d{3,4} to \d[\d,]{2,5} ms in this run", js) is None
    mem = js[js.index('detailFlow("memory"'):js.index('detailFlow("self_state"')]
    assert 'model_steps_per_s' in mem and "no throughput record" in mem and "snapshot steps you can page through" in mem



def test_p118_the_page_says_the_position_numbers_are_bounded_and_counts_the_pinned_ones():
    """Audit P118: the controller's position channel is bounded at its source since
    2026-09-19. The page must not describe the proposal as the network's plain unbounded
    output any more, must keep the measured before-numbers as history, and the motor
    panel must count the numbers pinned at the edge, since a bounded column can no longer
    show a runaway by its size."""
    from pathlib import Path
    from mimo_bridge.explainer.schema import DOC
    w = DOC["node.motor"]["w"]
    assert "held inside -1 to +1" in w and "plain output" not in w
    assert "791 of those 1,560" in w and "did not change" in w
    # the entry moved to term.candidate-command on 2026-10-02 (viewer.js DOC_ALIAS keeps the old key)
    c = DOC["term.candidate-command"]["w"]
    assert "580 of the H81 probe's 1,560" in c and "slope above 0" in c
    js = (Path(__file__).resolve().parents[1] / "explainer" / "viewer.js").read_text(encoding="utf-8")
    assert "position numbers pinned at the edge of the body's range" in js
    assert "proposed now: x,y,z (raw)" not in js


# ---------------------------------------------------------------------------------------
# Audit rows reviewed 2026-09-19 (viewer group): P392 P393 P449 P451 P212 P329 P041.
# The viewer is checked the way the rest of this file checks it - by reading the shipped
# source and by building a page in Python; no browser is available to the suite.
# ---------------------------------------------------------------------------------------
def _viewer_js():
    return open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()


def _viewer_css():
    return open(os.path.join(BX.PKG, "viewer.css"), encoding="utf-8").read()


# ================================================================= P392
def _node_param_map(js, name="NODE_PARAM"):
    """viewer.js NODE_PARAM, read out of the shipped source: box id -> the component_params
    keys its corner count may come from (a bare string is a list of one). With name="HOSTED_PARAM":
    box id -> the keys of the repair paths drawn as lines inside that box."""
    block = js[js.index("const %s = {" % name):]
    block = block[:block.index("};")]
    block = re.sub(r"//[^\n]*", "", block)          # a comment may quote key-like text
    return {k: re.findall(r'"([^"]+)"', v)
            for k, v in re.findall(r'(\w+):\s*(\[[^\]]*\]|"[^"]*")', block)}


def test_the_not_drawn_list_is_what_no_drawn_box_accounts_for():
    """P392: the footer's 'not drawn' list was a set difference against a literal array of
    eight component names plus one per sense. The shared output LayerNorm IS drawn (the
    ln_out box) and was on neither list, so a run built with
    encoder_defaults.output_layernorm_affine on would print its 256 numbers as 'not drawn'
    beside the box that draws them, and the box could never carry the count. The list is
    now whatever no drawn box accounted for."""
    from mimo_bridge.lem_model import LemLite
    js = _viewer_js()
    diagram = js[js.index("function buildDiagram()"):js.index("function buildLearnLoop()")]
    assert "const drawn = [" not in diagram, \
        "a literal list of component names decides what counts as drawn again"
    assert "direct.forEach(k => drawnComps.add(k));" in diagram
    assert "filter(([k]) => !drawnComps.has(k))" in diagram
    node_param = _node_param_map(js)
    assert node_param.get("ln_out") == ["enc_shared", "encoder_registry"], \
        "the shared LayerNorm box has no component key: its numbers reach neither its corner nor the drawn set"
    # behaviour, on the model's own receipt, with the norm's gain and bias switched on
    m = LemLite(device="cpu", seed=0, imagination=False,
                overrides={"encoder_defaults": {"output_layernorm_affine": True}})
    cp = m.receipt["component_params"]
    assert cp["enc_shared"] == 2 * m.d_model           # one gain and one bias per token number
    claimed = {k for keys in node_param.values() for k in keys}
    claimed |= {"enc_" + s for s in m.receipt["active_modalities"]} | {"forward_model"}
    leftover = set(cp) - claimed
    assert "enc_shared" not in leftover, "the footer would list the shared LayerNorm as not drawn"
    assert "gaze_head" in leftover, "the control: a part with no box must still be listed"
    assert "encoder_registry" in claimed, "the name the saved-tensor count files those numbers under"
    assert "output_layernorm_affine" in _page_doc()["node.ln_out"]["h"], \
        "the box's popup still states 'zero learned numbers' as a fact about every run"
    # THE TWO REPAIR PATHS DRAWN INSIDE THEIR HOST BOXES (the decided design of the Architecture tab, Oct 7, 2026,
    # D-E4): the motor encoder claims the asked-change path's key and the prediction heads the all-slot path's, by the
    # same rule, so the footer drops both; the liveness repair has no host box and stays listed (with its own popup)
    hosted = _node_param_map(js, "HOSTED_PARAM")
    assert hosted == {"enc_motor": ["motor_asked_change"], "heads": ["heads_all_slot_path"]}, hosted
    assert "extra.forEach(e => { if (e.claims) drawnComps.add(e.claims); });" in diagram
    h84 = {"enc_motor", "prediction_heads", "motor_asked_change", "heads_all_slot_path", "liveness_repair", "gaze_head"}
    claimed_now = claimed | {k for keys in hosted.values() for k in keys}
    assert h84 - claimed_now == {"liveness_repair", "gaze_head"}, h84 - claimed_now
    assert '"part.liveness_repair"' in diagram and "part.liveness_repair" in _page_doc()


# ================================================================= P393
def test_the_numbered_badges_take_no_clicks(tmp_path):
    """P393: badge() appends an r = 8 filled circle to the svg AFTER every box and line, so
    it is painted on top, and it carries no click handler. Without pointer-events: none it
    took the clicks over the top-left corner of every numbered box (35.3 square units of
    fill, 41.9 with its stroke) and over the stretch of the world-to-senses trunk line
    under each sense badge, against a legend that says any box or line can be clicked."""
    css = _viewer_css()
    rules = dict(re.findall(r"(\.dg-badge(?:-num)?)\s*\{([^}]*)\}", css))
    assert "pointer-events: none" in rules[".dg-badge"], \
        "the badge circle takes the click meant for the box corner and the line under it"
    assert "pointer-events: none" in rules[".dg-badge-num"]
    # why the rule is needed: appended to the svg itself, last, with no handler of its own
    js = _viewer_js()
    badge = js[js.index("  function badge(id, n) {"):]
    badge = badge[:badge.index("\n  }\n")]
    assert 'svg.appendChild(el("circle"' in badge and "addEventListener" not in badge
    # and the rule reaches the built page
    page = BX.render(_fixture_run(tmp_path))
    assert re.search(r"\.dg-badge\s*\{[^}]*pointer-events:\s*none", page)


# ================================================================= P449
def test_tile_5_names_its_snapshot_and_follows_the_inspectors_picker():
    """P449: tile 5's second sentence always read SNAPS[last] ('Newest snapshot: ...'),
    also on the inspector tab, where the row sits under the snapshot picker - so with
    step 0 selected it said 21 skin contacts above panels built from step 0's 18. It named
    neither the step nor, on a page whose charts are another run's, the run: the speed
    beside it is the chart run's and the snapshot is the probe's."""
    js = _viewer_js()
    tiles = js[js.index("function renderTiles("):js.index("const TABS = [")]
    assert "function renderTiles(selSnap) {" in tiles
    assert "const sN = selSnap || (SNAPS.length ? SNAPS[SNAPS.length - 1] : null);" in tiles
    assert '"The snapshot selected above"' in tiles and '"Newest snapshot"' in tiles
    assert '" (step " + sN.header.step' in tiles, "the sentence must name the step it describes"
    assert '(otherRun ? " of " + META.run : "")' in tiles, "and the run, when the charts are another run's"
    assert "`Newest snapshot: ${" not in tiles
    insp = js[js.index("function renderInspector()"):js.index("function renderRun()")]
    assert "root.appendChild(renderTiles(snap));" in insp
    assert insp.index("root.appendChild(sel);") < insp.index("root.appendChild(renderTiles(snap));"), \
        "'selected above' is true only while the picker row is appended before the tiles"
    run_tab = js[js.index("function renderRun()"):]
    assert "root.appendChild(renderTiles());" in run_tab, "the run tab has no picker: newest snapshot"
    assert "tile 5" in _page_doc()["tiles"]["w"]


# ================================================================= P451
def _drawable_nodes(receipt, args):
    """Mirror of viewer.js buildDiagram: every box id it draws for this receipt and these
    launch flags. A tour slide naming an id outside this set highlights nothing."""
    fm = receipt.get("forward_model", {})
    mods = receipt.get("active_modalities", [])
    stages = fm.get("stages") or {}
    boxes = {"world", "ln_out", "assembly", "final_norm", "heads", "recon", "motor", "gaze",
             "wellness", "curiosity", "intention", "imagination", "replay", "curriculum"}
    boxes |= {f"sense_{m}" for m in mods} | {f"enc_{m}" for m in mods}
    if (args.get("top_down_gate") or "off") != "off":
        boxes.add("td_gate")
    for i in range(int(fm.get("n_layers", 0) or 2)):
        for sid, key in (("s1", "cross_modal"), ("s2", "causal_temporal"), ("s3", "cross_modal_bridge")):
            if stages.get(key) is not False:
                boxes.add(f"block{i}_{sid}")
        boxes.add(f"block{i}_ffn")
    if (receipt.get("persistent_memory") or args.get("persistent_memory")) == "on":
        boxes.add("memory")
    if (receipt.get("self_state") or args.get("self_state") or "off") != "off":
        boxes.add("self_state")
    if (receipt.get("voice_head") or {}).get("enabled"):
        boxes.add("voice")
    if (args.get("hierarchical_self") or "off") != "off":
        boxes.add("hself")
    if (args.get("caregiver_mode") or "off") != "off":
        boxes.add("caregiver")
    if (args.get("entity_self") or "off") != "off":
        boxes.add("entity_self")
    for key, box in (("efference_heads", "efference"), ("contact_head", "contact_head"),
                     ("motor_synergy", "synergy")):
        if receipt.get(key):
            boxes.add(box)
    return boxes


def test_every_box_a_tour_slide_names_is_drawn(tmp_path):
    """P451: walkthrough.py falls back to the --self-state launch flag when the receipt
    carries no self_state field; buildDiagram read the receipt alone. On such a page the
    tour's self perception slide named a box the diagram never drew, and renderTour skips
    a missing box without a word. The edge check above covers lines; this covers boxes."""
    run = _fixture_run(tmp_path)
    args_path = os.path.join(run, "args.json")
    args = json.load(open(args_path))
    args["self_state"] = "learn"          # the launch flag; the fixture receipt has no self_state field
    json.dump(args, open(args_path, "w"))
    b = _blocks(BX.render(run))
    receipt = b["data-meta"]["receipt"]
    assert "self_state" not in receipt, "the fixture receipt must stay silent for this test to mean anything"
    assert b["data-meta"]["args"]["self_state"] == "learn", "the flag must reach the page for the diagram to read it"
    tour_nodes = {n for st in b["data-tour"] for n in st["nodes"]}
    assert "self_state" in tour_nodes, "the tour no longer falls back to the launch flag"
    missing = sorted(tour_nodes - _drawable_nodes(receipt, args))
    assert not missing, f"tour slides name boxes the diagram never draws: {missing}"
    # the mirror's self perception rule is the one viewer.js ships
    js = _viewer_js()
    diagram = js[js.index("function buildDiagram()"):js.index("function buildLearnLoop()")]
    rule = re.search(r"const selfMode = ([^;]+);", diagram)
    assert rule, "buildDiagram no longer names the mode it draws the self perception box from"
    assert "rec.self_state" in rule.group(1) and "argsm.self_state" in rule.group(1)
    tour = js[js.index("function renderTour()"):js.index("function snapByStep(")]
    assert "not drawn for this run" in tour, "a slide naming a missing box must say so on the page"


# ================================================================= P212 (residual)
def test_the_self_perception_panel_gives_one_measurement_one_row():
    """P212: consequence_error is the stats writer's alias for consequence_error_1 - the
    same float - and before the first executed order it is None while the 1-order key is
    absent. The panel now drops the alias, and a None anywhere prints as a dash."""
    import inspect
    from mimo_bridge import self_state_integration as SSI
    assert re.search(r"stats\['consequence_error'\]\s*=\s*stats\['consequence_error_1'\]",
                     inspect.getsource(SSI)), \
        "the alias is gone from the stats writer: drop the viewer's filter with it"
    js = _viewer_js()
    block = js[js.index("if (snap.self !== undefined)"):js.index("if (snap.hself !== undefined)")]
    assert "delete selfShown.consequence_error;" in block
    assert "labeledKv(selfShown, {" in block
    assert not re.search(r"^\s*consequence_error:", block, re.M), "the alias has a row of its own again"
    assert re.search(r"^\s*consequence_error_1:", block, re.M)
    kv = js[js.index("function labeledKv("):js.index("function renderInspector()")]
    assert 'v == null ? "—"' in kv, "a None field prints as the word null"


# ================================================================= P329
def test_the_page_says_which_output_the_motor_bar_grades():
    """P329: the motor term in the loss breakdown grades motor_for_loss - the controller
    run a second time on the PREVIOUS moment's stored inputs - while the order panel shows
    res.motor_out, this moment's new proposal. Two different outputs, and nothing on the
    page said so (H81 step 100: the column's largest position number 1.28, the bar 61,248)."""
    import inspect
    from mimo_bridge import lem_model as LM
    step_src = inspect.getsource(LM.LemLite.step)
    assert re.search(r"motor_for_loss, _lc, _lm = self\.motor_controller\(prev_lat, prev_hs,", step_src), \
        "the graded command is no longer the controller re-run on the lagged inputs: the page text must follow"
    assert re.search(r"loss_manager\.compute\(\s*predictions, targets, source_mask, motor_for_loss, motor_target",
                     step_src)
    doc = _page_doc()
    assert "is run a second time on the PREVIOUS model step's stored inputs" in doc["bp.grade"]["w"]
    assert "motor_for_loss" in doc["losses"]["h"] and "'proposed now' column" in doc["losses"]["f"]
    js = _viewer_js()
    insp = js[js.index("function renderInspector()"):js.index("function renderRun()")]
    shown = "\n".join(l for l in insp.splitlines() if not l.lstrip().startswith("//"))
    assert shown.count("PREVIOUS model step's stored inputs") >= 2, \
        "the command panel and the loss panel must both say which output the motor bar grades"
    assert "cannot be computed from this column" in shown


# ================================================================= P041 (residuals)
def test_the_stage_3_time_distance_chart_names_its_curves_and_what_an_older_snapshot_lacks():
    """P041 follow-up: the fix draws two curves, and lineChart draws no key, so nothing on
    the page said which line looks back and which looks forward. A snapshot written before
    the fix carries the backward curve alone and was drawn under a heading that said both
    directions. And lineChart put the run's restart, phase and snapshot markers on this
    chart, whose x axis counts moments of distance and not training steps."""
    js = _viewer_js()
    att = js[js.index("if (L.stage3_lag) {"):js.index("row.appendChild(wrap);")]
    assert 'el("div", { class: "legend" })' in att, \
        "two curves and no key: the reader cannot tell looking back from looking forward"
    assert "const both = Array.isArray(L.stage3_lag_forward);" in att
    assert "this snapshot carries no forward curve" in att
    assert "noStepMarkers: true" in att
    chart = js[js.index("function lineChart("):js.index("function docBtn(")]
    assert chart.count("stepAxis ?") == 3, "restart, phase and snapshot markers belong to step axes only"
    banned = "hon" + "est"          # spelled in two halves so this file does not contain the word
    assert not re.search(r"\b" + banned, js, re.I), "a word the project's writing rules ban is in the shipped source"
    s3 = _page_doc()["attention.stage3"]
    assert "LATER" in s3["w"] and "stage3_direction_share" in s3["h"]


# ======================================================================================
# audit rows P430 P431 P434 P443 P444 and the two findings under P369 (2026-09-19)
# ======================================================================================
import logging

import pytest


SIX = ["vision", "tactile", "audio", "proprioception", "motor", "intermind"]


def _chart(out, needle):
    hit = [c for c in out["charts"] if needle in c["label"]]
    return hit[0] if hit else None


def _keys(chart):
    return [s["key"] for s in chart["series"]]


def _six_sense_records(n=12):
    recs = []
    for i in range(n):
        recs.append({
            "type": "mimo", "step": 100 * (i + 1),
            "causal_pe": 0.02,
            "causal_pe_per_modality": {m: 0.01 + 0.001 * i for m in ("audio", "proprioception", "tactile", "vision")},
            "td_gate": {m: 0.1 + 0.001 * i for m in ("audio", "proprioception", "tactile", "vision")},
            "recon_dec": {m: 0.2 + 0.001 * i for m in ("audio", "proprioception", "tactile", "vision")},
            "pe": {m: 0.003 + 0.0001 * i for m in ("audio", "intermind", "proprioception", "tactile", "vision")},
            "token_temporal_std": {m: 0.4 + 0.001 * i for m in SIX},
        })
    return recs


# ------------------------------------------------------------------ P430
def test_p430_every_sense_the_log_carries_is_drawn_when_no_receipt_exists_yet():
    """The model receipt is written with the first checkpoint, --checkpoint-every-min (60)
    minutes after launch, so for the first hour of every run segment build_series gets
    modalities=None. The senses must then come from the log, not from a three-sense list."""
    out = BX.build_series({"mimo": _six_sense_records(), "training": [], "launch": []}, [])
    assert "causal_pe_per_modality.audio" in _keys(_chart(out, "causal error"))
    assert "recon_dec.audio" in _keys(_chart(out, "Reconstruction decoders"))
    assert {"pe.audio", "pe.intermind"} <= set(_keys(_chart(out, "Predicted change")))
    assert "td_gate.audio" in _keys(_chart(out, "Top-down gate"))
    # all six, and in the receipt's own order, so a sense keeps its line colour when the
    # first receipt arrives
    assert _keys(_chart(out, "collapse watch")) == ["token_temporal_std." + m for m in SIX]


def test_p430_a_sense_only_the_log_names_is_drawn_beside_the_receipt_s_list():
    recs = _six_sense_records()
    for r in recs:
        r["td_gate"]["vestibular"] = 0.05
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [], modalities=SIX)
    assert "td_gate.vestibular" in _keys(_chart(out, "Top-down gate"))


def test_p430_tile_4_searches_the_collapse_series_the_page_carries():
    js = _viewer_js()
    start = js.index("function renderTiles(")
    tiles = js[start:js.index("// ---", start)]
    assert '["vision", "tactile", "proprioception", "motor"]' not in tiles, \
        "tile 4 still falls back to a four-sense list when the page has no receipt"
    assert '"token_temporal_std."' in tiles


# ------------------------------------------------------------------ P431
def _training_records(n=10):
    return [{"type": "training", "training_step": 100 * (i + 1), "avg_loss": 1.0 / (i + 1),
             "loss_breakdown": {"next_frame": 0.5, "imagination": 0.3, "vicreg": 0.9,
                                "tenure_tok": 0.1}} for i in range(n)]


def test_p431_the_loss_terms_chart_is_drawn_from_records_keyed_by_training_step():
    """Live-step loss records carry training_step and no step (679 of 679 on H84)."""
    recs = _training_records()
    assert all("step" not in r for r in recs)
    out = BX.build_series({"mimo": [], "training": recs, "launch": []}, [])
    loss = _chart(out, "Loss terms")
    assert loss is not None, "the only chart of the teaching signal over a run is missing"
    # DESIGN.md section 11 (2026-09-23): each lesson is drawn as its share of the window's positive
    # lessons added up (a loss has no lived unit); the raw sum avg_loss is no longer a line
    assert _keys(loss) == ["loss_breakdown.next_frame.share", "loss_breakdown.imagination.share",
                           "loss_breakdown.vicreg.share", "loss_breakdown.tenure_tok.share"]
    assert loss["series"][0]["points"][-1][0] == 1000
    vic = next(s for s in loss["series"] if s["key"] == "loss_breakdown.vicreg.share")
    assert abs(vic["points"][0][1] - 0.9 / (0.5 + 0.3 + 0.9 + 0.1)) < 1e-6
    assert loss["unit"] == "/100"


def test_p431_a_declared_chart_no_record_can_fill_is_named(caplog):
    with caplog.at_level(logging.WARNING, logger="mimo_bridge.build_explainer"):
        out = BX.build_series({"mimo": _six_sense_records(), "training": [], "launch": []}, [])
    assert any("Loss terms" in lab for lab in out["dropped"])
    assert not any("collapse watch" in lab for lab in out["dropped"])
    assert any("Loss terms" in r.getMessage() for r in caplog.records)
    assert "SERIES.dropped" in _viewer_js(), "the run tab never prints the charts it could not draw"


# ------------------------------------------------------------------ P434
def _contract_records(n, fresh=True, contract=True):
    recs = []
    for i in range(n):
        r = {"type": "mimo", "step": 100 * (i + 1)}
        if contract:
            r["contract"] = {"exec_seen": 100 * (i + 1) - 1, "step": 100 * (i + 1)}
        if fresh:
            r["vision_fresh_fraction"] = 1.0
        recs.append(r)
    return recs


def test_p434_chart_6_is_the_contract_share_with_or_without_the_fresh_vision_field():
    stats = {"mimo": _contract_records(20, fresh=False), "training": [],
             "launch": [{"type": "launch", "step": 0}]}
    c6 = _chart(BX.build_series(stats, []), "executed command reach")
    assert c6 is not None, "20 contract records and no chart 6"
    assert _keys(c6) == ["exec_frac"]
    assert c6["trend"]["key"] in _keys(c6), "the trend chip names a series the chart does not carry"


def test_p434_fresh_vision_is_never_drawn_under_the_executed_order_title():
    stats = {"mimo": _contract_records(20, contract=False), "training": [], "launch": []}
    out = BX.build_series(stats, [])
    assert _chart(out, "executed command reach") is None
    fresh = [c for c in out["charts"] if "vision_fresh_fraction" in _keys(c)]
    assert len(fresh) == 1 and fresh[0]["doc"] == "chart.vision_fresh"
    for c in out["charts"]:
        if c.get("trend"):
            assert c["trend"]["key"] in _keys(c), c["label"]


def test_p434_the_contract_share_reaches_the_newest_record_of_a_long_run():
    n = BX.SERIES_POINT_CAP + 300
    stats = {"mimo": _contract_records(n), "training": [], "launch": [{"type": "launch", "step": 0}]}
    c6 = _chart(BX.build_series(stats, []), "executed command reach")
    ex = [s for s in c6["series"] if s["key"] == "exec_frac"][0]
    assert ex["points"][-1][0] == 100 * n, "the series stops before the newest record"


# ------------------------------------------------------------------ P443
def _tile_1(out):
    """Mirror of viewer.js renderTiles tile 1 (seriesByKey + winStats, no seams)."""
    def by_key(key):
        for ch in out["charts"]:
            for s in ch["series"]:
                if s["key"] == key:
                    return s["points"]
        return None

    def win(pts):
        if not pts or len(pts) < 2:
            return None
        k = max(1, min(100, len(pts) // 4))
        return sum(p[1] for p in pts[-k:]) / k
    m = re.search(r'const (\w+) = winStats\(seriesByKey\("(tenure_tok\.[a-z_]+)"\)\);\s*\n\s*if \(r1 && \1',
                  _viewer_js())
    assert m, "tile 1's second lookup moved; update this mirror"
    key = m.group(2)
    val = win(by_key(key))
    body = _viewer_js()[m.end():m.end() + 400]
    if "1 / (1 + " in body:                 # the value is treated as a decoy COUNT
        return key, (None if val is None else 1 / (1 + val))
    return key, val                          # the value is treated as the chance itself


def test_p443_tile_1_states_the_run_s_own_chance():
    recs = [{"step": i, "tenure_tok": {"rank1": 0.20, "n_decoys": 14, "vis_gate_open": 0.4}}
            for i in range(40)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    key, chance = _tile_1(out)
    assert chance is not None, f"tile 1 looks up {key}, which no chart carries"
    assert abs(chance - 1 / 15) < 1e-4, \
        f"tile 1 reads {key} and states guessing as {chance:.4f}; 14 decoys make it {1 / 15:.4f}"
    for ch in out["charts"]:
        for s in ch["series"]:
            if s["key"] == "tenure_tok.n_decoys":
                assert min(p[1] for p in s["points"]) >= 1, \
                    "a series keyed as the decoy COUNT holds values below 1"


def test_p443_speed_contacts_and_stillness_debt_do_not_share_an_axis():
    recs = [{"step": i, "movement": {"local_speed": 10.0 + i, "stillness_debt": 0.002 * (i % 3)},
             "contacts_per_env": 18.0 + 0.1 * i} for i in range(40)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    homes = {}
    for ch in out["charts"]:
        for s in ch["series"]:
            homes[s["key"]] = ch
    for key in ("movement.local_speed", "contacts_per_env", "movement.stillness_debt"):
        assert key in homes, f"{key} is no longer drawn anywhere"
        ch = homes[key]
        if ch["log"]:
            continue
        allv = [p[1] for s in ch["series"] for p in s["points"]]
        mine = [p[1] for s in ch["series"] if s["key"] == key for p in s["points"]]
        share = (max(mine) - min(mine)) / (max(allv) - min(allv))
        assert share > 0.25, f"{key} occupies {100 * share:.2f}% of its chart's height"


# ------------------------------------------------------------------ P444
def test_p444_a_series_near_one_millionth_keeps_its_resolution():
    raw = [1.502e-06 + i * 5.0e-09 for i in range(94)]
    recs = [{"step": i, "td_gate": {"vision": v}} for i, v in enumerate(raw)]
    s = BX._series(recs, "td_gate.vision", "vision")
    got = [p[1] for p in s["points"]]
    assert len(set(got)) >= 90, f"{len(set(raw))} distinct values were plotted as {len(set(got))} levels"
    assert max(abs(g - r) / r for g, r in zip(got, raw)) < 1e-5
    tiny = BX._series([{"step": 0, "x": 4.814e-07}, {"step": 1, "x": 4.9e-07}], "x", "x")
    assert all(p[1] > 0 for p in tiny["points"]), "a nonzero value was plotted as exactly 0.0"


def test_p444_a_value_that_is_not_finite_cannot_blank_the_run_tab():
    recs = [{"step": i, "causal_pe": 0.02} for i in range(6)]
    recs[3]["causal_pe"] = float("nan")
    recs[4]["causal_pe"] = float("inf")
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])

    def strict(_):
        raise ValueError("NaN or Infinity in the page's JSON: JSON.parse rejects the whole block")
    json.loads(json.dumps(out), parse_constant=strict)
    assert len(_chart(out, "causal error")["series"][0]["points"]) == 4


# ------------------------------------------------------------------ P369 (adjacent: --watch drops --stats-from)
def _bare_run(root, name):
    d = root / name
    d.mkdir(parents=True)
    (d / "args.json").write_text("{}")
    (d / "replay_training_stats.jsonl").write_text(json.dumps({"type": "launch", "step": 0}) + "\n")
    return str(d)


@pytest.mark.parametrize("which_log_grows", ["the --run directory's", "the --stats-from run's"])
def test_p369_a_watched_page_keeps_charting_the_stats_from_run(tmp_path, monkeypatch, which_log_grows):
    """Since 2026-09-27 (commit 520ef50a4) --watch runs explainer/refresh.py's refresh_loop: every rebuild is a fresh
    build_explainer.py child process behind the memory guard, never BX.build in this process. Until 2026-10-02 this
    test still patched BX.build and called main() without --out, so it took the LIVE page's refresh lock (it failed
    while the live refresher held that lock, and with the lock free it would have looped on the live page's path with
    the real sleep, which refresh_loop binds at definition). Here the loop runs two cycles on a page under tmp_path,
    with its guard and builder commands recorded and never started."""
    from types import SimpleNamespace
    from mimo_bridge.explainer import refresh as RF
    probe, training = _bare_run(tmp_path, "probe"), _bare_run(tmp_path, "training")
    grows = probe if which_log_grows.startswith("the --run") else training
    out = str(tmp_path / "explainer.html")
    children = []

    def command(cmd, **kwargs):
        if cmd[2].endswith("build_explainer.py"):
            children.append(cmd)
        return SimpleNamespace(returncode=0, stdout="allowed", stderr="")

    def grow(_seconds):
        with open(os.path.join(grows, "replay_training_stats.jsonl"), "a") as fh:
            fh.write(json.dumps({"type": "mimo", "step": 100}) + "\n")

    real = RF.refresh_loop
    monkeypatch.setattr(RF, "refresh_loop", lambda a, choose, stamp, target: real(
        a, choose, stamp, target, run_command=command, sleep=grow, max_cycles=2))
    assert BX.main(["--run", probe, "--watch", "--every", "2", "--stats-from", training, "--out", out]) == 0
    charted = [c[c.index("--stats-from") + 1] if "--stats-from" in c else None for c in children]
    assert charted == [training, training], \
        f"builds charted {charted}: the rebuild after {which_log_grows} log grew lost --stats-from, or never ran"
    assert all(c[c.index("--out") + 1] == os.path.abspath(out) for c in children), "the page under tmp_path, never the live one"


def test_p434_the_depth_licence_win_shares_reach_the_newest_record_of_a_long_run():
    n = BX.SERIES_POINT_CAP + 300
    recs = [{"type": "mimo", "step": 100 * (i + 1),
             "plan_license": {"licensed": 2, "win_emas": [0.6, 0.55, 0.4]},
             "intention": {"horizon": 3, "plan_length": 2}} for i in range(n)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    lic = _chart(out, "Depth licence")
    ends = {s["key"]: s["points"][-1][0] for s in lic["series"]}
    assert set(ends.values()) == {100 * n}, f"series on one chart end at different steps: {ends}"
    assert all(len(s["points"]) <= BX.SERIES_POINT_CAP + 1 for s in lic["series"])


def test_p369_every_popup_that_says_this_run_can_say_whose_run_it_means(tmp_path):
    from mimo_bridge.explainer import schema as SCH
    assert SCH.DOC_WRITTEN_FOR["run"] and SCH.DOC_WRITTEN_FOR["what"]
    run = tmp_path / "H99_other_run"
    run.mkdir()
    (run / "args.json").write_text("{}")
    meta = _blocks(BX.render(str(run)))["data-meta"]
    assert meta["doc_written_for"] == SCH.DOC_WRITTEN_FOR
    js = _viewer_js()
    assert "META.doc_written_for" in js and " - NOT " in js
    uses = sum(1 for v in SCH.DOC.values()
               if re.search(r"this run|this probe|the audited run", " ".join(map(str, v.values())), re.I))
    assert uses > 0, "no entry says 'this run' any more - the popup note and this test can go"


def test_viewer_js_is_syntactically_valid_javascript():
    """No test executes viewer.js, so one unbalanced quote ships a page whose every tab is
    blank. esprima (pure Python, already in .venv_mimo) parses the whole file."""
    esprima = pytest.importorskip("esprima")
    esprima.parseScript(_viewer_js())


# ======================================================================================
# audit rows P028 P050 P183 P272 P406, reviewed 2026-09-19 (explainer text group)
# ======================================================================================
def test_the_touch_text_says_which_parts_become_rows():
    """P028: the page said 'the 32 parts pressing hardest become touch rows'. The code admits
    every part with ANY nonzero field - a vibration strength stays nonzero for up to one second
    after the touch ends - then sorts by pressure and keeps 32. On the H81 probe's 12 embedded
    snapshots 94 of 190 rows had zero pressure, rub, slip and area."""
    import torch
    from mimo_bridge.lem_model import touch_point_cloud, TOUCH_TOKENS, TOUCH_TOKEN_F
    imp = torch.zeros(256, 10)
    imp[0, 0] = 0.9                      # a part pressed now
    imp[1, 4] = 0.3                      # no pressure, rub, slip or area: one vibration strength only
    out = touch_point_cloud(imp, None)
    assert int(out[:, TOUCH_TOKEN_F - 1].sum()) == 2, "a vibration-only part is no longer admitted - the page text must follow the code"
    assert float(out[1, 3]) == 0.0 and abs(float(out[1, 7]) - 0.3) < 1e-6
    imp = torch.zeros(256, 10)
    imp[:40, 0] = torch.linspace(0.05, 1.0, 40)
    imp[40:42, 4] = 0.5                  # 40 pressed parts + 2 vibration-only parts
    out = touch_point_cloud(imp, None)
    assert int(out[:, TOUCH_TOKEN_F - 1].sum()) == TOUCH_TOKENS and float(out[:, 3].min()) > 0, \
        "past the cap the low-pressure end is dropped, so the zero-pressure rows go first"
    doc = _page_doc()
    for key in ("node.sense.tactile", "edge.sense_tactile.enc"):
        text = " ".join(str(v) for v in doc[key].values())
        assert "pressing hardest become touch rows" not in text and "strongest-pressing 32" not in text, key
    assert "one second after a touch ends" in doc["node.sense.tactile"]["w"]
    assert "ended within the last second" in doc["edge.sense_tactile.enc"]["w"]
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    touch = js[js.index("// --- touch"):js.index("// --- attention")]
    assert "Each dot is one touched body part" not in touch and "active contact tokens" not in touch
    assert "pressedNow" in touch and "ended within the last second" in touch


def test_the_position_range_is_named_in_the_world_frame():
    """P050: the proposal's -1..+1 is 200 cm either side of the WORLD origin on each axis.
    The action decoder reads a target as 'cm, world', and the reach clamp adds the clamped
    proposal x 200 to the achieved WORLD pose x 200 - so the range is not centred on the
    body, whose middle sat 12 to 25 cm from the origin on the H81 probe's snapshots."""
    import inspect
    from mimo_bridge import action_decoder as AD, lem_model as LM
    assert "target position (cm, world)" in (AD.__doc__ or "")
    step = inspect.getsource(LM.LemLite.step)
    assert "cur_cm = handle_state[:, :, 0:3].detach().float() * 200.0" in step
    assert "des_cm = src_cmd.detach().float()[:, :, 0:3].clamp(-1.0, 1.0) * 200.0" in step
    w = _page_doc()["node.motor"]["w"]
    assert "body's own centre" not in w, "the page centres the position range on the body; the code uses the world origin"
    assert "WORLD's origin" in w


def test_the_memory_picture_names_what_the_recurrent_network_reads_and_how_often():
    """P183: the arrow into the GRU box carried no contract. The network reads each event plus
    two log-compressed time numbers, over the WHOLE retained window on every call, and step()
    makes that call twice per moment when the delayed next-frame lesson is on."""
    import inspect
    import math
    from lem.model.persistent_memory import PersistentMemory
    from mimo_bridge import lem_model as LM, memory_integration as MI
    pm = PersistentMemory(input_size=1140, n_modalities=6, d_model=128, action_size=234)
    n_in = pm.input_projection[0].in_features
    assert n_in == 1140 + 2, "the memory no longer joins exactly two time numbers to each event"
    assert "self.persistent_memory.encode(**self._nf_memory)" in inspect.getsource(LM.LemLite.step)
    assert "model.persistent_memory.encode(**snapshot,observe=True)" in inspect.getsource(MI.observe)
    e = _page_doc()["edge.memory.ring_gru"]
    # the row's width is this run's own since the H84 sync: the entry carries the
    # placeholder, and shape_fill's formula is held to the module built above
    assert "{memory_event_timed} numbers per row" in e["w"] and "twice per [[model-step]]" in e["w"]
    old_shape = {"encoders_built": {"vision": {"d_output": 128}},
                 "forward_model": {"_n_modalities": 6}, "handle_names": ["h"] * 26}
    assert BX.SCH.shape_fill(old_shape, {})["{memory_event_timed}"] == f"{n_in:,}"
    # DESIGN.md section 11: the squeeze is stated as the ratio it leaves, not as the raw numbers
    assert round(math.log1p(64) / math.log1p(0.5)) == 10 and "about ten times apart" in e["w"]
    assert "128 times apart" in e["w"] and 64 / 0.5 == 128
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    mem = js[js.index('detailFlow("memory"'):js.index('detailFlow("self_state"')]
    assert '"edge.memory.ring_gru"' in mem, "the arrow into the GRU box opens no data contract"
    assert "2 time numbers" in mem and "twice per model step" in mem


def test_the_hearing_scale_is_described_as_a_level_in_tenfold_steps():
    """P272: the page called the spectrum 'energy' and its steps 'decibels'. What the code maps to
    0-1 is 10 x log10 of an amplitude-like sum - each source's gain times its spectral shape, speech
    joining as the square root of its power - so the true statement is a level in tenfold steps."""
    import inspect
    from mimo_bridge import audio_adapter as AA
    src = inspect.getsource(AA.CribAudio.render)
    assert "speech_magnitude = np.sqrt(speech_power)" in src, "speech no longer joins as an amplitude"
    assert "db = 10.0 * np.log10(spec + 1e-9)" in src, "the 0-1 mapping changed: re-derive the page's scale text"
    step = 10.0 / AA.DB_SPAN                       # the 0-1 distance of one tenfold rise in level
    doc = _page_doc()
    scale_text = doc["term.frequency-band"]["w"]
    # DESIGN.md section 11: the step is written as its share of the scale, never as a bare decimal
    assert step == 0.125 and f"every eighth of the scale ({100 * step:g}%) below it is a tenfold drop in level" in scale_text
    assert "is ten decibels" not in scale_text
    for key in ("node.sense.audio", "edge.world.sense_audio"):
        assert "energy" not in doc[key]["w"], f"{key} calls an amplitude-like level an energy"
    assert f"{10 ** (AA.DB_SPAN / 10.0):,.0f} times" in doc["edge.world.sense_audio"]["w"]


def test_the_moment_length_comes_from_the_receipt_and_is_typed_nowhere(tmp_path):
    """P406: '50 ms' was typed into term.model-step (then keyed term.moment), term.motor-hold ('5 in current runs'), node.world,
    two places in walkthrough.py and nine in viewer.js, all resting on an unexported 0.01. The
    trainer now records the MEASURED env step in the receipt (env_dt_s, model_step_s); the page
    reads it, so a run at another --action-repeat or another scene timestep prints its own moment."""
    import inspect
    from mimo_bridge.explainer import walkthrough as WT
    run = _fixture_run(tmp_path)
    args_path = os.path.join(run, "args.json")
    args = json.load(open(args_path))
    args["action_repeat"], args["motor_hold"] = 3, 4           # not the crib's 5 and 5
    json.dump(args, open(args_path, "w"))
    rpath = os.path.join(run, "checkpoints", "ckpt_step_100", "model_receipt.json")
    rec = json.load(open(rpath))
    rec["env_dt_s"], rec["model_step_s"] = 0.02, 0.06          # a 20 ms env step, measured
    json.dump(rec, open(rpath, "w"))
    b = _blocks(BX.render(run))
    assert b["data-meta"]["clock"] == {"env_step_s": 0.02, "action_repeat": 3,
                                       "model_step_s": 0.06, "measured": True}
    doc = b["data-doc"]
    assert "3 physics steps of 20 ms" in doc["term.model-step"]["w"] and "60 ms of body time" in doc["term.model-step"]["w"]
    assert "measured on its own body" in doc["term.model-step"]["w"]
    assert "4 in H99_fixture" in doc["term.motor-hold"]["w"] and "240 ms" in doc["term.motor-hold"]["w"]
    assert "3 physics steps of 20 ms = 60 ms" in doc["node.world"]["when"]
    for k, v in doc.items():
        for kk, vv in v.items():
            if isinstance(vv, str):
                assert not re.search(r"\{(env_step_ms|env_step_source|action_repeat|moment_ms|motor_hold|hold_ms)\}", vv), (k, kk)
    loop = next(st for st in b["data-tour"] if st["id"] == "loop")
    assert "0.06 s of body time" in loop["body"]
    # a receipt written before the export falls back to the scene's own 0.005 s x 2, and says so
    rec.pop("env_dt_s"); rec.pop("model_step_s")
    json.dump(rec, open(rpath, "w"))
    old = _blocks(BX.render(run))
    clock = old["data-meta"]["clock"]
    assert clock["measured"] is False and abs(clock["model_step_s"] - 0.03) < 1e-12
    assert "assumed because this run's receipt" in old["data-doc"]["term.model-step"]["w"]
    assert "measured on its own body" not in old["data-doc"]["term.model-step"]["w"]
    # and no shipped source types the env step a second time
    assert "0.01 *" not in inspect.getsource(WT), "walkthrough.py computes the model step from a typed 0.01 again"
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    code = re.sub(r"^[ \t]*//.*$", "", js, flags=re.M)
    for literal in ("0.01 *", "* 0.01", "0.005 * 2", "10 * reps", "10 * (a.action_repeat",
                    '"every 0.05 s"', "one command = 0.05 s"):
        assert literal not in code, f"viewer.js still types the model step's length: {literal}"
    assert "delay[1] * 10" not in inspect.getsource(BX.phase_table)


# ---------------------------------------------------------------- audit P267 (2026-09-19)
def test_p267_the_caregiver_view_says_whose_motion_the_moving_toy_line_reports():
    """The adult says 'the toy is moving' when the toy shifted 1.5 cm in 0.5 s while the
    baby watches it. No contact can move the toy; on run H84 78 of 100 such lines followed
    the baby moving its own head, which the adult follows with the toy. The page must say
    so and must not say a contact puts no force on the toy."""
    from pathlib import Path
    from mimo_bridge.explainer.schema import DOC
    w = DOC["node.caregiver"]["w"]
    assert "a contact puts no force on it" not in w
    assert "a push cannot move it" in w and "78 of 100" in w and "688" in w
    js = (Path(__file__).resolve().parents[1] / "explainer" / "viewer.js").read_text(encoding="utf-8")
    flow = js[js.index('detailFlow("caregiver"'):js.index("let detailId = null;")]
    assert "a watched toy shifts 1.5 cm" in flow
    assert "a push cannot move it" in flow and "78 of 100" in flow
    notes = [ln.strip() for ln in flow.splitlines() if ln.strip().startswith('"')]
    assert max(len(n) for n in notes) <= 170, "a footnote is one unwrapped line in a 900-unit drawing"


# ---------------------------------------------------------------- audit P117 (2026-09-19)
def test_the_planner_text_quotes_the_model_s_own_numbers():
    """Audit P117. Every planner number the page prints is rebuilt here from the model
    that owns it, so lem_model.py cannot change a constant and leave the page quoting
    the old one. Three places restate them: schema.DOC, the tour's intention slide and
    the planner mini-diagram in viewer.js."""
    import inspect
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge.explainer import walkthrough as WT
    from lem.model.intention_generator import IntentionGenerator
    m = LemLite(device="cpu", seed=0, intention=True, intention_execute=False,
                imagination=False)
    ig = m.intention_gen
    n, lr, reg, ring = ig._n_steps, ig._lr, ig._strength_reg, m.goal_ring.maxlen
    fill = int(re.search(r"len\(self\.goal_ring\) >= (\d+) and step %",
                         inspect.getsource(LemLite.step)).group(1))
    doc = _page_doc()
    # DESIGN.md section 11 (2026-09-23): the page writes these two settings in words; the words are
    # held to the model's own values here
    words = {0.1: "one tenth", 0.01: "one hundredth"}
    assert lr in words and reg in words, f"re-word the page: lr {lr}, strength charge {reg}"

    # the planner text rewritten on 2026-10-02 to state input, computation and output in the canonical
    # names (a command, a model step, a goal-ring entry); the numbers are still the model's own
    quotes = [
        ("node.intention", "w", f"**Computation, {n} times**"),
        ("node.intention", "w", f"step size {words[lr]}"),
        ("node.intention", "w", f"plus {words[reg]} of the sum over the"),
        ("node.intention", "w", f"Of the {n + 1} commands scored (the starting command and the {n} updates)"),
        ("node.intention", "when", f"holds at least {fill} entries"),
        ("intention", "when", f"once the goal ring holds {fill} entries"),
        ("term.forward-model", "w", f"{n} Adam updates"),
        ("term.candidate-command", "w", f"the search's {n} updates"),
        ("term.predicted-outcome", "w", f"The {n} updates push"),
        ("term.predicted-outcome", "w", f"plus {words[reg]} of every handle's squared strength"),
        ("term.intention-distance", "w", f"{words[reg]} of the sum over the"),
        ("term.adam", "w", f"step size {words[lr]} on the command's numbers, {n} updates"),
        ("bp.forward", "w", f"the planner's {n} rounds"),
        ("term.goal-ring", "w", f"last {ring} model steps"),
        # the ring's length in SECONDS is this run's own since the H84 items (2026-09-19):
        # a placeholder here, filled from the run's measured model step (checked below)
        ("term.goal-ring", "w", "{goal_ring_s} seconds of body time at the {moment_ms} ms model step"),
        ("term.goal-ring-entry", "w", f"newest {ring} entries"),
    ]
    for key, field, text in quotes:
        assert text in doc[key][field], \
            f"{key}.{field} no longer quotes the model's own value: expected {text!r}"
    # the ring's length in seconds: the constant the page multiplies is the model's own,
    # and the product follows the run's clock (50 ms by default, 60 ms when it says so)
    assert BX.SCH.GOAL_RING_MOMENTS == ring
    # the hierarchical self's two windows are the class's own momenta, not typed seconds
    from mimo_bridge.hierarchical_self import HierarchicalSelfDark
    hs = HierarchicalSelfDark(["a"], 8, n_slots=4)
    for tag, moments in BX.SCH.HSELF_WINDOW_MOMENTS.items():
        assert moments == int(round(1.0 / (1.0 - hs.level_momentum[tag]))), tag
    when = BX.SCH.DOC["node.hself"]["when"]
    assert "{hself_l2_window_min}" in when and "{hself_l3_window_min}" in when
    assert "500 s" not in when and "8.3 minutes" not in when, "a typed length of a window"
    assert BX.SCH.run_fill({}, {}, "R")["{goal_ring_s}"] == f"{ring * 0.05:g}"
    assert BX.SCH.run_fill({"env_dt_s": 0.02, "model_step_s": 0.06}, {"action_repeat": 3},
                           "R")["{goal_ring_s}"] == f"{ring * 0.06:g}"
    assert "10 seconds of body time at the 50 ms step" not in doc["term.goal-ring"]["w"]

    # the tour's intention slide
    receipt = {"active_modalities": list(m.active),
               "forward_model": {"_seq_length": m.ctx_len * m.n_mod,
                                 "_n_modalities": m.n_mod, "n_layers": 2, "n_heads": 8},
               "handle_names": list(m.handle_names)}
    slide = next(s for s in WT.build_steps(receipt, {}) if s["id"] == "intention")["body"]
    assert f"the last {ring} model steps' tokens" in slide
    assert f"runs {n} Adam updates" in slide and f"step size {words[lr]}" in slide
    assert f"the lowest-scoring of the {n + 1} commands" in slide.lower()

    # the planner diagram (the intention box's detail panel), drawn on a grid since 2026-10-02
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    mini = js[js.index("function intentionMini("):js.index("function seriesByKey(")]
    assert f'"last {ring} model steps, per body:"' in mini
    assert f"back through the filter: {n} updates" in mini
    assert f"of {n + 1} scored commands" in mini and f"(start + {n} updates), per body" in mini
    assert f"step size 1/{round(1 / lr)}; weights fixed" in mini
    assert f"1/{round(1 / reg)} \u00d7 \u03a3 strength\u00b2" in mini

    # what the 'compared' count on the predicted-outcome arrow counts: the picture's
    # filter and the objective's filter are the same rule - not motor, measured live
    assert 'Object.keys(sm).filter(m => m !== "motor" && sm[m] === "real").length' in mini
    gen = inspect.getsource(IntentionGenerator.generate_action)
    assert 'if m != "motor" and m in desired_encoded' in gen
    assert 'self._source_mask.get(m, "real"))) == "real"}' in gen
    assert "Only senses measured live and not motor are compared" in doc["term.predicted-outcome"]["w"]


# ---------------------------------------------------------------- audit P318 (2026-09-19)
def test_the_grading_station_text_matches_the_code():
    """The shortened grading overview keeps timing and raw-target contracts.
    Preserve source lesson inventory, on/off weighting and replay-clock checks
    without requiring a historical run's long loss list in the overview popup."""
    import inspect
    from mimo_bridge.lem_model import LemLite
    from mimo_bridge import replay as RP
    from lem.training.loss_manager import LossManager
    doc = _page_doc()
    grade, replay = doc["bp.grade"]["w"], doc["bp.replay"]["w"]
    m = LemLite(device="cpu", seed=0, imagination=False, next_frame_loss=True)   # the runs' setting
    lm = m.loss_manager
    step = inspect.getsource(LemLite.step)
    # Retain the default-value regressions formerly coupled to the long prose.
    # They do not imply these defaults apply to the run selected in the viewer.
    sig = inspect.signature(LemLite.__init__).parameters
    assert (float(lm._vicreg_variance_weight), float(lm._vicreg_covariance_weight),
            float(lm._imagination_weight), float(lm._loss_weights["motor"]),
            float(lm._wellness_weight), float(lm._latent_reg_weight)) == (1.0, 0.04, 0.5, 1.0, 0.1, 0.01)
    assert tuple(float(sig[name].default) for name in
                 ("anticipation_gaze_w", "action_tenure_weight", "action_tenure_token_weight")) == (0.1, 0.5, 0.25)
    hi, lo = re.search(r"w_copy = float\(min\(([\d.]+), max\(([\d.]+),", step).groups()
    assert (float(lo), float(hi)) == (0.02, 0.3)
    mem_w = re.search(r"loss = loss \+ (\.?\d*\.?\d+)\*memory_loss", step).group(1)
    assert float(mem_w) == 0.1
    assert "same command and time" in grade
    assert "Other enabled lessons have their own targets, weights and schedules" in grade
    assert "recorded loss terms for the selected snapshot" in grade
    # Keep the two unweighted lesson additions as source invariants.
    assert "loss = loss + nf_loss" in step
    assert "loss = loss + self_state_loss" in step
    # --next-frame-loss zeroes the five per-sense prediction weights; off, they are these
    heads = [k for k in lm._loss_weights if k != "motor"]
    assert all(lm._loss_weights[k] == 0.0 for k in heads) and len(heads) == 5
    off = LemLite(device="cpu", seed=0, imagination=False,
                  next_frame_loss=False).loss_manager._loss_weights
    assert (float(off["vision"]), float(off["tactile"]), float(off["audio"]),
            float(off["proprioception"]), float(off["intermind"])) == (1.0, 0.5, 0.8, 1.0, 0.0), \
        f"the off-flag prediction weights changed ({off})"

    # 2. Preserve the source inventory of breakdown keys, including optional
    #    lessons. The overview sends readers to the selected snapshot's terms.
    keys = set(re.findall(r"""bd\[["'](\w+)["']\]\s*=""", step))
    keys |= set(re.findall(r'breakdown\["(\w+)"\](?:\[\w+\])?\s*=',
                           inspect.getsource(LossManager.compute)))
    named = {"vicreg": "VICReg", "imagination": "the imagination chain",
             "recon": "the reconstruction guard", "next_frame": "the delayed next-model-step guess",
             "anticipation_gaze": "anticipation-gaze", "motor": "the motor lesson",
             "persistent_memory": "persistent memory's guess", "wellness": "[[wellness]] copy",
             "latent_reg": "the latent regulariser",
             "self_state_loss": "self perception's present-and-forecast lesson",
             "tenure": "the tenure pose hinge", "tenure_tok": "command-to-outcome matching term",
             "prediction": "per-sense prediction terms"}
    not_in_these_runs = {
        "gaze": "gaze_target is None at the only call site",
        "voice": "voice_target is None at the only call site",
        "cross_sim": "no cross-simulation loss is passed",
        "motor_mse": "--wellness-mode forward only", "motor_wellness": "--wellness-mode forward only",
        "contact_consequences": "--contact-learning on only"}
    assert keys == set(named) | set(not_in_these_runs), (
        f"loss breakdown keys the page's list does not account for: "
        f"{sorted(keys - set(named) - set(not_in_these_runs))}; stale entries here: "
        f"{sorted((set(named) | set(not_in_these_runs)) - keys)}")
    assert "gaze_target=None, voice_out=voice_out, voice_target=None" in step

    # 3. which tokens the live per-sense term reads: the raw ones, both lines (P042)
    assert "current={m: tokens_raw[m].squeeze(1) for m in tokens_raw}," in step
    assert "targets = {m: tokens_raw[m].squeeze(1).detach()" in step
    assert "raw sensed targets" in doc["bp.loop"]["w"]
    assert "Targets use current raw encodings detached from the gradient" in doc["bp.forward"]["w"]

    # 4. re-study's numbers, from the three places that own them
    tm = open(os.path.join(os.path.dirname(inspect.getfile(BX)), "..", "train_mimo.py"),
              encoding="utf-8").read()
    every = int(re.search(r'"--replay-every", type=int, default=(\d+)', tm).group(1))
    batch = int(re.search(r'"--replay-batch", type=int, default=(\d+)', tm).group(1))
    size = int(re.search(r'"--replay-size", type=int, default=(\d+)', tm).group(1))
    assert "if len(starts_tok) < max(2, batch // 4):" in inspect.getsource(RP)
    # the cadence, the batch and the skip line are THIS RUN's flags since the H84 sync: the
    # entry carries placeholders, and shape_fill falls back to the launcher's own defaults
    # for a run that set neither flag
    assert "every {replay_every} model steps" in replay
    assert "requests {replay_batch} windows of {replay_window} model steps" in replay
    assert "Fewer than {replay_floor} usable windows skips the pass" in replay
    unset = BX.SCH.shape_fill({}, {})
    assert unset["{replay_every}"] == str(every) and unset["{replay_batch}"] == str(batch)
    assert unset["{replay_floor}"] == str(max(2, batch // 4))
    assert float(lm._imagination_weight) == 0.5
    # the ring's size is THIS RUN's --replay-size since the H84 items (2026-09-19): both
    # entries carry the placeholder, a run that recorded the flag prints its own value, and
    # only a run that did not falls back to the launcher's default - and says it did
    assert "a ring of {replay_size} model steps per body" in doc["edge.bp.replay.update"]["w"]
    assert "the ring holds {replay_size} model steps per body" in doc["edge.ln_out.replay"]["h"]
    assert BX.SCH.REPLAY_SIZE_DEFAULT == size
    assert BX.SCH.run_fill({}, {"replay_size": 1234}, "R")["{replay_size}"] == "1,234"
    unrecorded = BX.SCH.run_fill({}, {}, "R")["{replay_size}"]
    assert unrecorded.startswith(f"{size:,} (the launcher's default")
    default_every = int(re.search(r'"--intention-every", type=int, default=(\d+)', tm).group(1))
    assert BX.SCH.INTENTION_EVERY_DEFAULT == default_every


# ---------------------------------------------------------------- audit P417 / P418 / P425 (2026-09-19)
# "lem_model.py:1412", "lem_model.py line 1412", "viewer.js:665" ...
_FILE_LINE = re.compile(r"[A-Za-z_]+\.(?:py|js|yaml)(?::| lines? )\d+")
# ... and a bare ":1378" or ":2744-2750" that continues such a citation
_BARE_LINE = re.compile(r"(?<![\w.\d]):\d{3,4}(?:-\d{3,4})?\b")


def _line_citations(text):
    return _FILE_LINE.findall(text) + _BARE_LINE.findall(text)


def test_the_page_cites_code_by_name_never_by_line_number():
    """Audit P417/P418/P425: a line number is wrong the next time anyone edits the file
    above it. The top-down gate's blend was published as 'lem_model.py line 1412'; it
    stood at line 1492 on 2026-09-18 and at 1586 on 2026-09-19. Counted on 2026-09-19,
    the page text and the explainer package's comments carried 20 line numbers for
    lem_model.py, self_state_integration.py, self_state.py and vision_adapter.py, and
    17 of the 20 pointed at a line that no longer held the named code. Page text and
    explainer comments name the function and quote the statement instead;
    test_the_anchors_the_page_names_exist checks that each named anchor still exists."""
    from mimo_bridge.explainer import walkthrough as WT
    offenders = []
    for key, entry in _page_doc().items():
        for field, text in entry.items():
            for hit in _line_citations(str(text)):
                offenders.append(f"DOC {key}.{field}: {hit}")
    mods = ["vision", "tactile", "audio", "proprioception", "motor", "intermind"]
    receipt = {"active_modalities": mods, "prediction_head_modalities": mods[:4] + ["intermind"],
               "forward_model": {"_seq_length": 768, "_n_modalities": 6, "n_layers": 2, "n_heads": 8},
               "handle_names": [f"h{i}" for i in range(23)] + ["gaze", "lower_trunk", "chest"],
               "persistent_memory": "on", "self_state": "learn",
               "voice_head": {"enabled": True, "params": 93264}}
    args = {"top_down_gate": "on", "hierarchical_self": "dark", "hself_slots": 6400,
            "caregiver_mode": "on", "audio": "real", "plan_horizon": "earned",
            "curiosity_goal": "on", "motor_state_formula": "boundary"}
    for st in WT.build_steps(receipt, args):
        for hit in _line_citations(st["body"]):
            offenders.append(f"tour slide {st['id']}: {hit}")
    for name in ("viewer.js", "capture.py", "walkthrough.py", "writer.py", "__init__.py"):
        src = open(os.path.join(BX.PKG, name), encoding="utf-8").read()
        for n, line in enumerate(src.splitlines(), 1):
            hits = _FILE_LINE.findall(line)
            mark = "//" if name.endswith(".js") else "#"
            if mark in line:                       # a bare :NNN only counts inside a comment
                hits += _BARE_LINE.findall(line.split(mark, 1)[1])
            offenders += [f"{name} line {n}: {h}" for h in hits]
    assert not offenders, "line-number citations that will drift:\n  " + "\n  ".join(offenders)


def test_the_anchors_the_page_names_exist():
    """The other half: a named anchor can go stale too, so each function and each
    quoted statement the rewritten entries lean on is looked up in the live code."""
    import inspect
    from mimo_bridge import lem_model as LM, self_state_integration as SSI
    from mimo_bridge import vision_adapter as VA
    from lem.model import self_state as SS
    step = inspect.getsource(LM.LemLite.step)
    init = inspect.getsource(LM.LemLite.__init__)
    # term.goal-ring-entry: the blend is written into the dict the goal ring later stores
    blend = "tokens[m] = tokens[m] + (g.unsqueeze(-1) * (zp.unsqueeze(1) - tokens[m]))"
    ring = "self.goal_ring.append(({m: tokens[m].squeeze(1).detach().float() for m in tokens},"
    assert step.index("tokens_raw = {m: tokens[m] for m in tokens}") < step.index(blend) < step.index(ring)
    # bp.update: the optimizer, its two groups, and the split clip
    assert "self.optimizer = torch.optim.AdamW(" in init
    assert "decay = [p for p in params if p.dim() >= 2]" in init
    assert "clip_grad_norm_(original,self.grad_clip)" in inspect.getsource(LM.LemLite._clip_training_gradients)
    # the self-perception entries
    obs = inspect.getsource(SSI.observe)
    assert "context=torch.cat([tokens[m].detach().float()" in obs
    assert "observed=torch.cat([sites[...,-3:]," in obs
    assert "target=observed.detach()" in obs and "history.events.append(dict(" in obs
    assert "target=torch.stack([e['target'] for e in events],1)" in obs
    assert "action=command_for_encoder(executed.detach().float())" in obs
    assert "h=self.history(self.command(" in inspect.getsource(SS.GroundedDynamics.forward)
    assert "loss = loss + self_state_loss" in step
    assert "self.self_history.internal = last_step.detach().float().mean(1)" in step
    assert "if self_state_bias is not None and self.self_state_mode == 'perceive':" in step
    # the viewer's comments: memory observes before the gate, the controller writes the
    # order, the dark observer reads raw tokens inside the same step
    assert step.index("from .memory_integration import observe") < step.index(
        "if self.top_down_gate and self._nf_pred is not None:")
    assert "motor_out, motor_conf, motor_meta = self.motor_controller(" in step
    assert 'self._hself.observe(self.last_tokens, wellness_out["grounded"])' in step
    # capture.py's docstring
    assert "laid out row-major" in inspect.getsource(VA.image_to_patches)


# ---------------------------------------------------------------- H84 sync (2026-09-19)
def _fixture_run_at_shape(tmp_path, d_model=192, n_blocks=4):
    """_fixture_run with its receipt rewritten to a model launched at --d-model 192
    --core-blocks 4: every encoder's d_output, the block count, and a per-tensor inventory
    of the core at that width. The context stays the fixture's 4 moments."""
    run = _fixture_run(tmp_path)
    rpath = os.path.join(run, "checkpoints", "ckpt_step_100", "model_receipt.json")
    rec = json.load(open(rpath))
    d, ctx = d_model, 4
    for m in rec["encoders_built"]:
        rec["encoders_built"][m]["d_output"] = d
    rec["forward_model"]["n_layers"] = n_blocks
    rec["core_blocks"] = n_blocks

    def par(*shape):
        n = 1
        for s in shape:
            n *= s
        return {"shape": list(shape), "kind": "parameter", "elements": n}

    tens = {"modality_embedding.weight": par(6, d), "temporal_embedding.weight": par(ctx, d),
            "final_norm.weight": par(d), "final_norm.bias": par(d)}
    for b in range(n_blocks):
        tens[f"blocks.{b}.causal_mask"] = {"shape": [ctx, ctx], "kind": "buffer",
                                           "elements": ctx * ctx}
        for stage in ("cm", "ct", "cmb"):
            tens[f"blocks.{b}.{stage}_norm.weight"] = par(d)
            tens[f"blocks.{b}.{stage}_attn.in_proj_weight"] = par(3 * d, d)
        tens[f"blocks.{b}.ffn_norm.weight"] = par(d)
        tens[f"blocks.{b}.ffn.0.weight"] = par(4 * d, d)
    learned = sum(t["elements"] for t in tens.values() if t["kind"] == "parameter")
    saved = sum(t["elements"] for t in tens.values())
    rec["component_params"]["forward_model"] = learned
    rec["total_params"] = sum(rec["component_params"].values())
    inv = rec["checkpoint_inventory"]
    inv["optimizer_parameters"] = rec["total_params"]
    inv["components"]["forward_model"] = {"parameters": learned,
                                          "saved_tensor_elements": saved, "tensors": tens}
    json.dump(rec, open(rpath, "w"))
    return run, {"block": 3 * (d + 3 * d * d) + d + 4 * d * d, "learned": learned,
                 "saved": saved}


def test_the_page_reads_the_model_s_size_from_the_receipt_and_types_none_of_it(tmp_path):
    """H84 sync (2026-09-19). DOC, the chart captions and viewer.js were written about a
    128-wide, 2-block model and typed its size as fact: "a token is 128 numbers", "[6 x
    128]", "768 numbers", "1,140 numbers", "this model stacks 2 such blocks", "5,728,601".
    The size is a launch setting now (--d-model --core-blocks ...; H84 runs 192 wide with 4
    blocks), so the page has to read it from the run's own receipt: schema.model_shape is
    the one source, shape_fill fills the prose, META.shape feeds viewer.js."""
    run, core = _fixture_run_at_shape(tmp_path / "wide", d_model=192, n_blocks=4)
    b = _blocks(BX.render(run))
    doc, meta = b["data-doc"], b["data-meta"]

    # (1) the glossary's token entry states THIS run's width
    token = doc["term.token"]["w"]
    assert "a token is 192 numbers" in token, token
    assert "128" not in token, "the glossary still types the old model's token width"

    # (2) no placeholder survives anywhere in the built page's text: not one of
    # shape_fill's own, and nothing else shaped like one
    rec = json.load(open(os.path.join(run, "checkpoints", "ckpt_step_100", "model_receipt.json")))
    args = json.load(open(os.path.join(run, "args.json")))
    keys = set(BX.SCH.shape_fill(rec, args))
    assert "{d_model}" in keys and "{n_blocks}" in keys and "{token_row}" in keys
    texts = [(k, kk, vv) for k, v in doc.items() for kk, vv in v.items() if isinstance(vv, str)]
    texts += [("chart", ch.get("label"), ch["caption"]) for ch in b["data-series"]["charts"]
              if isinstance(ch.get("caption"), str)]
    for k, kk, vv in texts:
        left = [p for p in keys if p in vv] + re.findall(r"\{[a-z][a-z0-9_]*\}", vv)
        assert not left, (k, kk, left)
    # ... and the unfilled DOC does carry them, so a typed number cannot pass as "filled"
    assert "{d_model}" in BX.SCH.DOC["term.token"]["w"]
    assert "{n_blocks}" in BX.SCH.DOC["bp.backward"]["w"]

    # (3) META carries the shape block viewer.js reads, from the receipt
    shape = meta["shape"]
    assert shape["d_model"] == 192 and shape["core_blocks"] == 4
    assert shape["source"]["d_model"] == "receipt" and shape["source"]["core_blocks"] == "receipt"
    assert shape["token_row"] == 6 * 192 and shape["context_length"] == 4 and shape["seq_len"] == 24
    # the code's own formulas: (senses + 1) x width + 10 + handles x 9 (lem_model.py),
    # width + 3 (lem/config/receipt.py), and the per-tensor inventory for the core
    assert shape["memory_event"] == 7 * 192 + 10 + 26 * 9
    assert shape["intermind_row"] == 192 + 3
    assert shape["core_block_learned"] == core["block"]
    assert shape["core_learned"] == core["learned"] and shape["core_saved"] == core["saved"]
    assert f"{core['block']:,} learned numbers" in doc["node.core"]["h"]
    assert "this model stacks 4 such blocks" in " ".join(doc["bp.backward"]["w"].split())
    assert f"{shape['memory_event']:,} numbers" in doc["edge.ln_out.memory"]["w"]
    assert "[6 x 192]" in doc["edge.final_norm.heads"]["w"]
    assert any("192-number token" in json.dumps(step) for step in b["data-tour"]),         "the tour does not state this run's token width"

    # the same build on the unchanged 128-wide, 2-block fixture says 128 and 2: the number
    # is READ, not replaced by another typed one
    b2 = _blocks(BX.render(_fixture_run(tmp_path / "narrow")))
    assert "a token is 128 numbers" in b2["data-doc"]["term.token"]["w"]
    assert b2["data-meta"]["shape"]["d_model"] == 128
    assert b2["data-meta"]["shape"]["core_blocks"] == 2

    # (4) viewer.js types none of it any more
    js = _viewer_js()
    assert "const SHAPE = META.shape || {};" in js
    for typed in ("d_output || 128", "× 128]", "[1140]", "n_layers || 2", "128→80",
                  "GRU, 128 per body", "context_length || 128", "131 placeholder",
                  "128 goal numbers", "the same 128 numbers", "768 raw sense-token numbers"):
        assert typed not in js, f"viewer.js still types {typed!r}"


# ======================================================================================
# H84 items, second pass (2026-09-19): the twelve things the first sync found and did not
# edit, plus the page's size. Numbering follows docs/explainer_h84_sync_2026-09-19/
# CLASSIFICATION.md. The rule for all of them: the page states THIS run's facts, read from
# its own receipt, args.json, stats log and checkpoints when it is built.
# ======================================================================================
# every string the older runs' pages typed as a fact and the unfilled DOC must not carry
_OLD_TYPED = ("262,022", "912,143", "162,950", "99,072", "4,097,248", "650,121",
              "within 10 places", "still do not", "three-layer", "this run's value, 5",
              "with both flags at 5 - this", "--intention-every 5 dividing",
              "5 and 5 in this run", "latched every 5 model steps", "one step in five",
              "for five steps", "about 50 ms of body time", "at the 50 ms step",
              "0.25 s of body time", "adds 3 more transitions", "3 steps back in this run",
              "--imagination-steps, 3 here", "Cap 3 (--imagination-steps)",
              "a ring of 6,000 model steps", "the ring holds 6,000 model steps",
              "none, in this run", "was none of the steps the body received",
              "as in this run", "5 in current runs")


def _set_args(run, **changes):
    path = os.path.join(run, "args.json")
    args = json.load(open(path))
    args.update(changes)
    json.dump(args, open(path, "w"))
    return args


def _edit_receipt(run, fn):
    path = os.path.join(run, "checkpoints", "ckpt_step_100", "model_receipt.json")
    rec = json.load(open(path))
    fn(rec)
    json.dump(rec, open(path, "w"))
    return rec


def test_items_4_to_7_the_flag_values_and_moment_lengths_are_the_runs_own(tmp_path):
    """Items 4, 5, 6 and 7. The page typed the H81 probe's flag values as facts of "this
    run": --intention-every 5 with --motor-hold 5, a 50 ms moment, an imagination depth of
    3 and a replay ring of 6,000. A run launched with other values must print its own, and
    the typed strings must be gone from the text the page is built from."""
    flat = " ".join(str(v) for e in _page_doc().values() for v in e.values())
    for old in _OLD_TYPED:
        assert old not in flat, f"the page's source text still types {old!r}"
    run = _fixture_run(tmp_path)
    _set_args(run, intention="on", intention_every=7, motor_hold=4, action_repeat=3,
              replay_size=1234, imagination="on", imagination_steps=2)
    _edit_receipt(run, lambda r: r.update(env_dt_s=0.02, model_step_s=0.06))
    doc = _blocks(BX.render(run))["data-doc"]
    when = doc["node.intention"]["when"]
    assert "On a model step whose number is a multiple of 7" in when
    assert "7 is the value H99_fixture was launched with" in when
    assert "latched for the body every 4 model steps" in when and "held for 240 ms of body time" in when
    assert "holds at least 20 entries" in when
    assert "latched every 4 model steps and held for 240 ms" in doc["edge.motor.world"]["w"]
    assert "12 env steps of 20 ms in H99_fixture" in doc["edge.motor.world"]["w"]
    assert "one command for 4 model steps (240 ms of body time in H99_fixture)" in doc["node.gaze"]["when"]
    assert "one model step in 4 in H99_fixture" in doc["node.sense.motor"]["f"]
    # item 5: every moment length follows the run's measured clock (60 ms here)
    assert "12 seconds of body time at the 60 ms model step of H99_fixture" in doc["term.goal-ring"]["w"]
    assert "60 ms of body time in H99_fixture" in doc["edge.bp.forward.grade"]["w"]
    assert "60 seconds of body time" in doc["term.collapse"]["h"]
    assert "60 s of body time at 3 physics sub-steps" in doc["chart.collapse"]["h"]
    assert "0.06 s of simulated body time at the 3 physics sub-steps of H99_fixture" \
        in doc["node.self_state"]["when"]
    # item 6: the imagination depth
    assert "The imagination chain adds 2 more transitions every model step" in doc["bp.forward"]["w"]
    assert "(2 steps back in H99_fixture)" in doc["node.imagination"]["w"]
    assert "The cap is --imagination-steps, 2 in H99_fixture." in doc["edge.imagination.intention"]["w"]
    assert "The cap is --imagination-steps, 2 in H99_fixture." in doc["node.intention"]["h"]
    # item 7: the replay ring
    assert "a ring of 1,234 model steps per body" in doc["edge.bp.replay.update"]["w"]
    assert "the ring holds 1,234 model steps per body" in doc["edge.ln_out.replay"]["h"]
    # a launch that recorded no depth says so; one with the chain off says that
    _set_args(run, imagination_steps=None)
    fwd = _blocks(BX.render(run))["data-doc"]["bp.forward"]["w"]
    assert "did not record --imagination-steps" in fwd and "adds 3 more" not in fwd
    _set_args(run, imagination="off")
    fwd = _blocks(BX.render(run))["data-doc"]["bp.forward"]["w"]
    assert "switched off in H99_fixture (--imagination off)" in fwd
    # a sentence that states a filled value names its run and never says "this run": the
    # viewer's popup note tells the reader that phrase means the run DOC was written about
    fill = BX.SCH.run_fill({}, {"intention": "on", "intention_every": 10, "motor_hold": 5}, "RUNQ")
    for key, val in fill.items():
        assert not re.search(r"this run|this probe|this page's run", val), (key, val)


def test_item_4_the_planner_sentences_follow_the_two_flags(tmp_path):
    """Item 4. Who writes the order the body executes changes with the two values, not
    only the numbers in the sentence: when --intention-every equals or divides
    --motor-hold every latch is a planner moment and exploration noise reaches the body
    only when the planner produced nothing; when it is a multiple (10 and 5, H84) every
    second latch has no plan and the noise executes there; when neither divides (7 and 5,
    H82) most plans miss the latch and are dropped."""
    run = _fixture_run(tmp_path)

    def built(every, hold, **more):
        changes = dict(intention="on", intention_every=every, motor_hold=hold)
        changes.update(more)
        _set_args(run, **changes)
        return _blocks(BX.render(run))["data-doc"]

    # the planner's own sentences (node.intention, edge.intention.motor) use the canonical names since
    # 2026-10-02 (a command, a latch step, a planner step); the motor controller's keep theirs
    h84 = built(10, 5)
    assert "every plan lands on a latch step" in h84["node.intention"]["when"]
    assert "Of every 2 latched commands 1 falls on a planner step and 1 does not" \
        in h84["node.intention"]["when"]
    assert "executed command on 1 of every 2 command boundaries" in h84["node.motor"]["when"]
    assert "This line moves the body on 1 of every 2 executed commands" in h84["edge.intention.motor"]["w"]
    assert "(1 of every 2 latched commands in H99_fixture" in h84["edge.motor.world"]["w"]
    assert "which is 1 of every 2 latched commands" in h84["term.executed-command"]["w"]
    assert "never authored an executed command" not in h84["node.motor"]["when"]

    h81 = built(5, 5)
    assert "every latch step is also a planner step" in h81["node.intention"]["when"]
    assert "carries [[babble]] noise only when the planner produced nothing" in h81["node.intention"]["when"]
    assert "writes an executed command only when the planner produced nothing" in h81["node.motor"]["when"]
    assert "none of the latched commands in H99_fixture" in h81["edge.motor.world"]["w"]

    h82 = built(7, 5)
    assert "1 plan in 5 lands on a latch step" in h82["node.intention"]["when"]
    assert "the other 4 are dropped" in h82["node.intention"]["when"]
    assert "the other 6 do not" in h82["node.intention"]["when"]

    off = built(10, 5, intention="off")
    assert off["node.intention"]["when"].startswith("H99_fixture WAS LAUNCHED WITH --intention off")
    assert "every executed command is this controller's proposal" in off["node.motor"]["when"]
    noexec = built(10, 5, intention="on", no_intention_execute=True)
    assert "no plan is ever sent to the body" in noexec["node.intention"]["when"]
    # the scheduled postural hold is live only under two flags; H84 has --rest-control choice
    live = built(10, 5, no_intention_execute=False, babble_bouts="real", rest_control="scheduled")
    assert "launched that way, so it is live there" in live["edge.motor.world"]["w"]
    choice = built(10, 5, babble_bouts="real", rest_control="choice")
    assert "--rest-control choice, so it does not happen there" in choice["edge.motor.world"]["w"]
    assert "so it does not happen there" in choice["term.executed-command"]["when"]


def test_item_4_the_executed_order_tally_is_the_runs_own_measurement(tmp_path):
    """Item 4. The two flags give the SCHEDULE; what happened is the trainer's tally at
    each latch (exec_source in the stats log). The page states that tally for the run the
    charts come from, and says unmeasured when the log carries none."""
    recs = [{"type": "mimo", "step": 100 * (i + 1), "elapsed_s": 60.0 * (i + 1),
             "exec_source": {"planner": 0.5, "reflex_noise": 0.3 + 0.01 * (i % 2),
                             "reflex": 0.2 - 0.01 * (i % 2)}} for i in range(10)]
    s = BX.exec_source_summary(recs)
    assert s["windows"] == 10 and s["first_step"] == 100 and s["last_step"] == 1000
    assert list(s["shares"]) == ["planner", "reflex_noise", "reflex"]
    assert s["shares"]["planner"] == {"mean": 0.5, "min": 0.5, "max": 0.5}
    assert abs(s["shares"]["reflex_noise"]["mean"] - 0.305) < 1e-9
    # only the windows after the log's last restart count: one log can hold two launches
    recs[6]["elapsed_s"] = 5.0
    for r in recs[7:]:
        r["elapsed_s"] = 5.0 + r["step"]
    assert BX.exec_source_summary(recs)["windows"] == 4
    assert BX.exec_source_summary([{"type": "mimo", "step": 1}]) is None
    said = BX.SCH.exec_source_text(s, "RUNS")
    assert "stats log of RUNS (10 report windows, steps 100 to 1,000)" in said
    assert "the planner wrote 50 of every 100" in said
    assert "0 of every 100 would mean that author never wrote an executed command" in said
    assert "unmeasured" in BX.SCH.exec_source_text(None, "RUNS")
    # on a built page: the fixture's log carries no tally, so the page says unmeasured
    run = _fixture_run(tmp_path)
    b = _blocks(BX.render(run))
    assert b["data-meta"]["exec_source"] is None
    assert "unmeasured on this page" in b["data-doc"]["node.motor"]["when"]


def _self_state_receipt(rec, d=16):
    """Give the fixture receipt a self-perception inventory: a legacy forecast head, the
    belief-into-perception bias, and one group a lesson does reach."""
    shapes = {"action_encoder.weight": [d, 10], "transition.weight_ih": [3 * d, d],
              "future.0.weight": [d, 2 * d], "residual.weight": [6 * d, d],
              "residual.bias": [6 * d], "grounded.net.weight": [d, d]}
    tens = {}
    for k, s in shapes.items():
        n = 1
        for x in s:
            n *= x
        tens[k] = {"shape": s, "kind": "parameter", "elements": n}
    total = sum(t["elements"] for t in tens.values())
    rec.update(self_state="learn", self_prediction="grounded_residual")
    rec["checkpoint_inventory"]["components"]["self_state"] = {
        "parameters": total, "parameter_tensors": len(tens), "tensors": tens}
    rec["component_params"]["self_state"] = total
    return {k: s for k, s in shapes.items()}


def _write_ckpt(run, step, shapes, residual_value=0.0, legacy_seed=0):
    import numpy as np
    from safetensors.numpy import save_file
    rng = np.random.default_rng(legacy_seed)
    out = {}
    for k, s in shapes.items():
        if k.startswith("residual."):
            out["self_state." + k] = np.full(s, residual_value, dtype=np.float32)
        else:
            out["self_state." + k] = rng.normal(size=s).astype(np.float32)
    d = os.path.join(run, "checkpoints", f"ckpt_step_{step}")
    os.makedirs(d, exist_ok=True)
    save_file(out, os.path.join(d, "model.safetensors"))


def test_item_1_the_idle_weight_sentence_is_this_runs_and_is_measured(tmp_path):
    """Item 1. "262,022 of its 912,143 weights ... still exactly zero after 4,097,248
    steps" was H83's count and H83's measurement, printed on every later run's page. The
    counts are the receipt's; whether the weights stayed put is measured on the run's own
    checkpoints each time the page is built, and reads unmeasured when there is none."""
    run = _fixture_run(tmp_path)
    shapes = {}
    _edit_receipt(run, lambda r: shapes.update(_self_state_receipt(r)))
    legacy = 16 * 10 + 48 * 16 + 16 * 32
    bias = 96 * 16 + 96
    total = legacy + bias + 16 * 16

    # no checkpoint file at all: the counts are stated, the measurement says unmeasured
    f = _blocks(BX.render(run))["data-doc"]["node.self_state"]["f"]
    assert f"{legacy + bias:,} of its {total:,} weights receive no gradient" in f
    assert f"{legacy:,} are the retired legacy forecast head" in f
    assert f"{bias:,} are the belief-into-perception bias" in f
    assert f.count("unmeasured: the run folder held no readable checkpoint") == 2
    assert f"{total - legacy - bias:,} weights are reached and trained" in f
    assert "H99_fixture" in f and "this run" not in f

    # one checkpoint: the zero check can be made, the comparison across time cannot
    _write_ckpt(run, 100, shapes)
    b = _blocks(BX.render(run))
    f = b["data-doc"]["node.self_state"]["f"]
    assert f"measured on the checkpoint at step 100: 0 of those {bias:,} numbers are nonzero" in f
    assert "the run folder holds one checkpoint (step 100)" in f

    # two checkpoints, nothing moved
    _write_ckpt(run, 200, shapes)
    b = _blocks(BX.render(run))
    f = b["data-doc"]["node.self_state"]["f"]
    m = b["data-meta"]["idle_measured"]
    assert m["step"] == 200 and m["first_step"] == 100
    assert m["groups"]["belief_bias"] == {"elements": bias, "nonzero": 0, "differ_from_first": 0}
    assert m["groups"]["legacy_head"]["differ_from_first"] == 0
    assert f"at steps 100 and 200: 0 of those {legacy:,} numbers differ between the two" in f
    assert "so they still hold the zero they were created with" in f
    groups = b["data-meta"]["idle_params"]["self_state"]["groups"]
    assert all(g["measured"].startswith("measured on the checkpoint") for g in groups)

    # THE CONTROL: weights that DID move are reported as moved, never as still zero
    _write_ckpt(run, 300, shapes, residual_value=0.25, legacy_seed=7)
    f = _blocks(BX.render(run))["data-doc"]["node.self_state"]["f"]
    assert f"{bias:,} of those {bias:,} numbers are nonzero, so something has moved them" in f
    assert "numbers differ between the two, so something has moved them" in f
    assert "still hold the zero" not in f

    # --self-state perceive: the bias IS used, so it is not listed as idle at all
    _edit_receipt(run, lambda r: r.update(self_state="perceive"))
    f = _blocks(BX.render(run))["data-doc"]["node.self_state"]["f"]
    assert "belief-into-perception bias" not in f and f"{legacy:,} of its {total:,}" in f
    # the viewer prints the measurement beside each group and names only the idle groups
    js = _viewer_js()
    assert 'g.why + (g.measured ? " - " + g.measured : "")' in js
    assert "its retired legacy forecast head and its belief-into-perception bias, both" not in js
    assert "its weights are still exactly zero" not in open(
        os.path.join(os.path.dirname(BX.PKG), "tools", "build_explainer.py"), encoding="utf-8").read()


def test_item_2_the_reach_arithmetic_is_held_to_the_real_vision_encoder():
    """Item 2. schema.swin_dependency repeats SwinBlock.forward's window arithmetic so the
    page can state the reach for ANY layer count. This holds it to the real encoder the way
    the reach was first measured: perturb ONE patch with a random vector (the LayerNorm in
    front of the attention erases a constant) and read which outputs of the Swin blocks
    moved. Every probed patch, every depth from 1 to 4, position for position."""
    import numpy as np
    import torch
    from lem.model.encoders.swin_patch import SwinPatchEncoder
    SCH = BX.SCH
    n = SCH.vision_patch_layout({"fovea_size": 288})[-1]["stop"]
    assert n == 1160 and SCH.vision_patch_layout({"fovea_size": 0})[-1]["stop"] == 512
    for layers in (1, 2, 3, 4):
        torch.manual_seed(layers)
        enc = SwinPatchEncoder(n_patches=n, patch_dim=12, d_internal=16, d_output=16,
                               patch_size=16, n_swin_layers=layers, n_heads=2,
                               window_size=SCH.SWIN_WINDOW, angular_pos_hidden=8,
                               ffn_expansion=2, dropout=0.0, bias=True, norm_type="layernorm",
                               pool_n_heads=2, patch_subsample=4096,
                               angular_pos_inputs=4).double().eval()
        dep = SCH.swin_dependency(n, SCH.SWIN_WINDOW, layers)
        x = torch.randn(1, n, 12, dtype=torch.float64)
        for patch in (0, 2, 3, 6, 7, 303, 511, 512, 1157, 1159):
            bump = torch.zeros_like(x)
            bump[0, patch] = torch.randn(12, dtype=torch.float64)
            with torch.no_grad():
                moved = (enc.backbone_forward(x) - enc.backbone_forward(x + bump)).abs().amax(-1)[0] > 0
            assert np.array_equal(moved.numpy(), dep[:, patch]), \
                f"{layers} layers, patch {patch}: the encoder and the arithmetic disagree"
    # the two figures the vision popup's 'how it is computed' text dates and names
    reach = {L: SCH.vision_reach({}, {"vision_swin_layers": L})["reach"] for L in (1, 2, 3, 4, 6)}
    assert reach == {1: 6, 2: 10, 3: 13, 4: 17, 6: 24}
    assert "10 places at 2 layers, 17 at 4" in SCH.DOC["node.enc.vision"]["h"]
    # the odd blocks roll the PADDED list and nothing masks the seam: the list's last 2
    # patches share a window with its first 3 (1,160 pads to 1,162; the shift is 3)
    assert SCH.vision_reach({}, {"vision_swin_layers": 2})["wrap"] == {"last": 2, "first": 3}
    assert SCH.vision_reach({}, {"vision_swin_layers": 1})["wrap"] is None


def _vision_layers(rec, n_layers):
    tens = rec["checkpoint_inventory"]["components"].setdefault(
        "encoder_registry", {"tensors": {}})["tensors"]
    for k in [k for k in tens if k.startswith("encoders.vision.blocks.")]:
        del tens[k]
    for i in range(n_layers):
        tens[f"encoders.vision.blocks.{i}.attn.in_proj_weight"] = {
            "shape": [48, 16], "kind": "parameter", "elements": 768}
    rec["encoders_built"]["vision"] = {"type": "swin_patch", "d_output": 128, "d_internal": 96}


def test_item_2_the_page_states_the_reach_for_this_runs_layer_count(tmp_path):
    """Item 2. "within 10 places", "the sharp centre's pairs still do not meet" and "no
    patch ever meets the patch above it" were measured on the 2-layer encoder. H84 has 4:
    the reach is 17 places and the sharp centre's pairs DO meet, while up-and-down
    neighbours still never do. The page computes it from the receipt's layer count."""
    run = _fixture_run(tmp_path)
    _edit_receipt(run, lambda r: _vision_layers(r, 4))
    b = _blocks(BX.render(run))
    w = b["data-doc"]["node.enc.vision"]["w"]
    assert b["data-meta"]["vision_reach"]["layers"] == 4
    assert b["data-meta"]["vision_reach"]["layers_source"] == "receipt"
    assert "4 layers deep" in w and "96 numbers wide inside" in w
    assert "for the 4 layers in the model receipt of H99_fixture" in w
    assert "within 17 places of it in that list of 1,160" in w
    assert "about 7 patches and 14 places apart, do meet: for every one of the 634" in w
    assert "so no patch ever meets the patch above it inside the layers" in w
    assert "its last 2 patches share a window with its first 3" in w
    tour = next(st for st in b["data-tour"] if st["id"] == "encoders")["body"]
    assert "at this run's 4 layers one patch reaches another only within 17 places" in tour

    _edit_receipt(run, lambda r: _vision_layers(r, 2))
    w2 = _blocks(BX.render(run))["data-doc"]["node.enc.vision"]["w"]
    assert "within 10 places of it" in w2
    assert "14 places apart, do not meet: for none of the 634" in w2

    _edit_receipt(run, lambda r: _vision_layers(r, 6))
    w6 = _blocks(BX.render(run))["data-doc"]["node.enc.vision"]["w"]
    assert "within 24 places" in w6 and "a patch meets the patch above it on" in w6
    # no depth recorded anywhere: the launcher's default, and the sentence says so
    bare = BX.SCH.vision_reach_text(BX.SCH.vision_reach({}, {}), "RUNB")
    assert "the launcher's default, because neither the receipt nor the launch flags of RUNB" in bare


def test_items_3_and_11_each_encoder_states_its_own_depth_and_width(tmp_path):
    """Items 3 and 11. --encoder-layers L does not build L layers everywhere (train_mimo.py
    size_overrides): tactile and motor get L attention layers, audio 2 + L convolutions,
    proprioception 1 + L hidden layers and an output layer. Each popup states what was BUILT,
    counted from the receipt's tensor inventory; H84's receipt gives the figures below."""
    def par(*shape):
        n = 1
        for s in shape:
            n *= s
        return {"shape": list(shape), "kind": "parameter", "elements": n}

    def h84_like(rec):
        t = {}
        for i in (0, 4, 8, 12):
            t[f"encoders.proprioception.mlp.{i}.weight"] = par(192, 312 if i == 0 else 192)
        for i in (1, 5, 9):
            t[f"encoders.proprioception.mlp.{i}.weight"] = par(192)      # a LayerNorm, not a layer
        for i in (0, 3, 6, 9):
            t[f"encoders.audio.spectral_conv.{i}.weight"] = par(192, 2 if i == 0 else 192, 3)
        t["encoders.audio.spectral_proj.weight"] = par(192, 192)
        for m in ("tactile", "motor"):
            for i in (0, 1):
                t[f"encoders.{m}.layers.{i}.attn.in_proj_weight"] = par(576, 192)
        t["encoders.intermind.layers.0.attn.in_proj_weight"] = par(384, 128)
        for i in range(4):
            t[f"encoders.vision.blocks.{i}.attn.in_proj_weight"] = par(576, 192)
        rec["checkpoint_inventory"]["components"]["encoder_registry"] = {"tensors": t}
        for m in rec["encoders_built"]:
            rec["encoders_built"][m]["d_internal"] = 128 if m == "intermind" else 192

    run = _fixture_run(tmp_path)
    rec = _edit_receipt(run, h84_like)
    d = BX.SCH.encoder_depths(rec)
    assert (d["vision_layers"], d["tactile_attn_layers"], d["motor_attn_layers"],
            d["audio_conv_layers"], d["proprio_linear_layers"], d["intermind_attn_layers"]) \
        == (4, 2, 2, 4, 4, 1)
    assert d["proprio_hidden_widths"] == [192, 192, 192]
    doc = _blocks(BX.render(run))["data-doc"]
    p = doc["node.enc.proprioception"]["w"]
    assert "a 4-layer [[mlp|MLP]] (3 hidden layers, each 192 numbers wide, then one output layer" in p
    assert "three-layer" not in p
    assert "2 [[attention]] layers over the rows, 192 numbers wide inside" in doc["node.enc.tactile"]["h"]
    assert "2 [[attention]] layers over the handle rows, 192 numbers wide" in doc["node.enc.motor"]["h"]
    assert "4 1-D convolutions in the spectrum branch, 192 numbers wide" in doc["node.enc.audio"]["h"]
    assert "1 self-[[attention]] layer, 128 numbers wide inside" in doc["node.enc.intermind"]["h"]
    for key in ("node.enc.tactile", "node.enc.motor", "node.enc.audio", "node.enc.intermind"):
        assert "H99_fixture" in doc[key]["h"]
    # a receipt with no tensor inventory types no count: it says the receipt carries none
    bare = _blocks(BX.render(_fixture_run(tmp_path / "bare")))["data-doc"]
    assert "a layer count the model receipt of H99_fixture does not carry" in bare["node.enc.tactile"]["h"]
    assert not re.search(r"\d-layer \[\[mlp", bare["node.enc.proprioception"]["w"])


def test_item_8_the_memory_label_is_this_pages_measurement_or_names_h81():
    """Item 8. "a third to a half the size of that token" was H81's measurement (0.19 to
    0.28 against 0.59) drawn on every run's memory diagram; H84's snapshots read 3.0 to 4.0
    for the same bias. The label now carries what the page's newest snapshot measured, the
    ratio when the snapshot records the landing token's size, and H81's figure only in a
    footnote that names H81."""
    js = _viewer_js()
    mem = js[js.index("const MEM_SNAP ="):js.index('detailFlow("self_state"')]
    assert "landing_token_rms" in mem and "bias_rms_body" in mem and "MEM_BIAS / MEM_TOKEN" in mem
    assert "l2: MEM_BIAS_L2" in mem
    assert 'l2: "[" + SHAPE.n_senses + " × " + D_MODEL + "], a third to a half' not in js
    drawn = [l for l in mem.splitlines() if not l.lstrip().startswith("//")]
    for line in drawn:
        if "a third to a half" in line:
            assert "H81" in line, "H81's figure is drawn without naming H81: " + line.strip()[:90]
    assert "that token's size is not in this snapshot" in mem
    panel = js[js.index('panel("Memory beyond the window"'):js.index('panel("Self perception"')]
    assert "landing_token_rms:" in panel and "bias_rms_body:" in panel
    assert "0.59 = as large as this lineage's own tokens" not in panel


def test_item_13_the_page_keeps_itself_inside_its_size_budget(tmp_path, monkeypatch):
    """Item 13. The builder only WARNED when a page went over its budget, and a snapshot's
    size follows the model's: 12 of H84's put the live page at 9.1 MB against 6 MB. The
    builder now embeds as many snapshots as fit - always the newest and the first, the rest
    spread evenly - and the page says how many exist and how many it carries."""
    run = _fixture_run(tmp_path, n_snap=10)
    idx_path = os.path.join(run, "explain", "index.json")
    idx = json.load(open(idx_path))
    for e in idx["snapshots"]:                      # pad every snapshot to about 80 KB
        p = os.path.join(run, "explain", e["file"])
        snap = json.load(open(p))
        snap["pad"] = "x" * 80000
        json.dump(snap, open(p, "w"), separators=(",", ":"))
        e["bytes"] = os.path.getsize(p)
    json.dump(idx, open(idx_path, "w"))
    rest = len(BX.render(_fixture_run(tmp_path / "plain", n_snap=0)).encode())
    budget_mb = (rest + 5.5 * 80500) / 1e6         # room for 5 of the 10, not for 6
    monkeypatch.setattr(BX, "PAGE_BUDGET_MB", budget_mb)
    page = BX.render(run)
    assert len(page.encode()) <= budget_mb * 1e6, "the page is over the budget it was given"
    b = _blocks(page)
    steps = [s["header"]["step"] for s in b["data-snapshots"]]
    assert len(steps) == 5 and steps[0] == 0 and steps[-1] == 9000, steps
    assert steps == [0, 2000, 5000, 7000, 9000], "the kept snapshots are not spread evenly"
    sn = b["data-meta"]["snapshots"]
    assert sn == {"on_disk": 10, "embedded": 5, "embedded_steps": steps, "limited_by": "budget",
                  "budget_mb": budget_mb, "max_snapshots": 12}
    assert b["data-meta"]["n_snapshots_embedded"] == 5 and b["data-meta"]["n_snapshots_on_disk"] == 10
    assert "5 of 10 snapshots embedded (the first and the newest always" in page
    # the markers on the charts still name every snapshot the run has
    assert b["data-series"]["snapmarks"] == [k * 1000 for k in range(10)]
    # a budget only the newest fits: it is kept, alone
    monkeypatch.setattr(BX, "PAGE_BUDGET_MB", (rest + 1.5 * 80500) / 1e6)
    one = _blocks(BX.render(run))["data-snapshots"]
    assert [s["header"]["step"] for s in one] == [9000]
    # room for everything: nothing is thinned and the header says so without a reason
    monkeypatch.setattr(BX, "PAGE_BUDGET_MB", 6.0)
    page = BX.render(run)
    assert "10 of 10 snapshots embedded ·" in page
    assert _blocks(page)["data-meta"]["snapshots"]["limited_by"] is None
    capped = _blocks(BX.render(run, max_snapshots=4))["data-meta"]["snapshots"]
    assert capped["embedded"] == 4 and capped["limited_by"] == "max_snapshots"
    assert capped["embedded_steps"] == [0, 3000, 6000, 9000]
    # the inspector says it beside its picker
    insp = _viewer_js()
    insp = insp[insp.index("function renderInspector()"):insp.index("function renderRun()")]
    assert "META.snapshots" in insp and "SN.on_disk > SN.embedded" in insp
    assert BX._even_indices(14, 7) == [0, 2, 4, 7, 9, 11, 13]
    assert BX._even_indices(14, 1) == [13] and BX._even_indices(3, 12) == [0, 1, 2]


def test_the_plan_row_reads_the_field_the_model_writes():
    """Found while checking the built H84 page: lem_model.py writes the launch setting as
    intention_execute_flag, the inspector went on reading `executed`, and all 14 of H84's
    snapshots printed the word "undefined" in the plan panel with nothing raised."""
    import inspect
    from mimo_bridge import lem_model as LM
    step = inspect.getsource(LM.LemLite.step)
    assert '"intention_execute_flag": bool(self.intention_execute)' in step
    # a dict KEY "executed" (a value on the same line), not the `== "executed":` of the
    # --smooth-source branch the synergy build added
    assert not re.search(r'"executed":[ \t]*\S', step), (
        "the model writes `executed` again: simplify the viewer's row")
    js = _viewer_js()
    assert "it.last.intention_execute_flag !== undefined ? it.last.intention_execute_flag : it.last.executed" in js
    # THE NEW TRUTH (gain-of-function build, 2026-09-22, DESIGN.md D13). The model still writes
    # None when it makes the plan - it cannot know whether the trainer will latch it - and with
    # --causal-chain on the trainer's CausalChainRecorder stamps the field at the latch: True
    # when the plan's first action was latched as the body's order that step, False when it was
    # computed and never sent (mimo_bridge/causal_chain.py, _on_latch). With the flag off the
    # field stays None. The page text that says nothing ever fills it
    # (explainer/schema.py, the plan panel's "when" paragraph) is the explainer agent's to
    # rewrite; this pin moves to the code that fills it.
    assert '"reached_body": None' in step
    from mimo_bridge import causal_chain as CC
    rec = inspect.getsource(CC.CausalChainRecorder._on_latch)
    assert 'li["reached_body"] = reached' in rec
    assert 'reached = bool(latched and plan is not None and source == "planner")' in rec
    trainer = open(os.path.join(os.path.dirname(BX.PKG), "train_mimo.py"), encoding="utf-8").read()
    assert "causal.on_latch(model, step, latched=latched_now" in trainer, \
        "the trainer no longer hands the latch to the recorder that stamps reached_body"


# ---------------------------------------------------------------- item 9: the page, executed
def _browser_or_skip():
    from mimo_bridge.tools import check_explainer_page as CK
    exe = CK.find_browser()
    if not exe:
        pytest.skip("no Edge, Chrome or Chromium on this machine (set EXPLAINER_BROWSER)")
    return CK, exe


def test_item_9_the_built_page_runs_in_a_headless_browser(tmp_path):
    """Item 9. No test executed viewer.js; esprima shows it parses and nothing more. This
    opens the BUILT page in a headless Chromium-family browser (--disable-gpu: the GPU
    belongs to the training run), switches every tab, opens every popup, clicks every box,
    draws every sub-diagram and tour slide and pages through every snapshot, and fails on
    an uncaught error, an exception, or the text "undefined", "[object Object]", a bare
    NaN or an unfilled {placeholder} anywhere a reader would see it."""
    CK, exe = _browser_or_skip()
    run = _fixture_run(tmp_path)
    _set_args(run, top_down_gate="on", hierarchical_self="dark", hself_slots=64,
              caregiver_mode="on", persistent_memory="on", self_state="learn",
              intention="on", intention_every=10, imagination="on", imagination_steps=3)
    page = tmp_path / "page.html"
    page.write_text(BX.render(run), encoding="utf-8", newline="")
    rep = CK.check_page(str(page), exe)
    assert rep["ran"], rep.get("why")
    assert not rep["problems"], "\n".join(rep["problems"])
    # four tabs since 2026-09-22: the probes and liveness tab joined the three
    assert all(t["button"] and t["section"] for t in rep["tabs"]) and len(rep["tabs"]) == 4
    assert rep["probes_tab_drawn"]
    assert rep["diagram_boxes"] >= 35 and rep["diagram_lines"] >= 40
    assert rep["learning_loop_drawn"] == 1 and rep["run_charts"] >= 12 and rep["run_tiles"] >= 3
    # the popups a reader opens are the map the suite checks: as many, and entry for entry the same
    # (a rewrite by the page's script leaves the count as it was; doc_changed sees it)
    assert rep["popups_opened"] == rep["doc_entries"] == len(_page_doc())
    assert rep["doc_changed"] == [], rep["doc_changed"]
    assert rep["boxes_clicked"] == rep["diagram_boxes"] and rep["sub_diagrams"] >= 10
    assert rep["tour_rendered"] == rep["tour_slides"] >= 14
    assert rep["snapshots_rendered"] == rep["snapshots"] == 3
    # 11 inspector panels: the nine that were always drawn, plus the two added on
    # 2026-09-21 - what the body map calls its own (UEMA 11.2) and the named things the
    # self-model watches. The fixture snapshot carries the blocks all eleven need, so a
    # panel that stops drawing - or one that draws "undefined" into the checker's own text
    # scan - fails here and not only in the source-text guards further down this file.
    assert rep["inspector_panels"] >= 11, rep["inspector_panels"]


def test_item_9_the_browser_check_fails_on_a_broken_page(tmp_path):
    """The controls. A check that cannot fail shows nothing, and the first version of this
    one could not: it scanned an element's whole textContent, which joins
    <b>plan sent to the body</b><span>undefined ...</span> into "bodyundefined", and the
    word-boundary match found nothing on a page that printed the word seven times."""
    CK, exe = _browser_or_skip()
    good = BX.render(_fixture_run(tmp_path))
    tail = "</script>\n</body></html>"
    assert good.endswith(tail)

    def check(name, page):
        p = tmp_path / (name + ".html")
        p.write_text(page, encoding="utf-8", newline="")
        return CK.check_page(str(p), exe)

    # an uncaught ReferenceError in the page's script
    start = good.rindex("<script>") + len("<script>")
    rep = check("reference_error", good[:start] + "notAFunctionAnywhere();\n" + good[start:])
    assert rep["ran"] and any("notAFunctionAnywhere is not defined" in p for p in rep["problems"])
    # a syntax error: the script never runs, and the report still arrives and says so
    rep = check("syntax_error", good[:start] + "function ((( {\n" + good[start:])
    assert rep["ran"] and any("SyntaxError" in p for p in rep["problems"])
    # text no reader should see, in an element whose neighbour's text abuts it. The
    # listener is put on WINDOW: a document listener runs before the page's own boot
    # handler, which then clears the run tab and the planted text with it
    body = good[:-len(tail)] + ';window.addEventListener("DOMContentLoaded", function () {' \
        'var b = document.createElement("b"); b.textContent = "plan sent to the body";' \
        'var s = document.createElement("span"); s.textContent = String(({}).missing) + " - a field the model renamed";' \
        'var r = document.getElementById("runtab"); r.appendChild(b); r.appendChild(s); });' + tail
    rep = check("undefined_text", body)
    assert rep["ran"] and any(p.startswith("undefined shown in run tab") for p in rep["problems"]), rep["problems"]
    # popup entries the page's own script writes at load, the way viewer.js wrote node.efference
    # and edge.efference.heads until 2026-09-25: a rewritten entry leaves the count of entries as it
    # was, so only the comparison with the embedded map can see it. Written at the end of the
    # page's script, after DOC exists and before the load handler runs.
    body = good[:-len(tail)] + ';DOC["term.token"] = {t: "rewritten at load"};' \
        'DOC["term.not_in_the_map"] = {t: "added at load"};' + tail
    rep = check("doc_written_at_load", body)
    assert rep["ran"] and sorted(rep["doc_changed"]) == ["term.not_in_the_map added",
                                                         "term.token rewritten"], rep.get("doc_changed")
    assert any(p.startswith("the page's script changed 2 popup entries") for p in rep["problems"]), \
        rep["problems"]
    # a popup rewritten only when it opens, after the load-time comparison: the exercise repeats the
    # comparison at its end (review of 2026-09-25, finding 6; the reviewer planted the same rewrite
    # in renderDocPanel and the load-time comparison alone reported nothing)
    body = good[:-len(tail)] + ';var _plantedRender = renderDocPanel; renderDocPanel = function () {' \
        'var t = docTrail[docTrail.length - 1];' \
        'if (t && t.key === "term.token") DOC["term.token"] = {t: "rewritten when opened"};' \
        'return _plantedRender(); };' + tail
    rep = check("doc_rewritten_when_opened", body)
    assert rep["ran"] and rep["doc_changed"] == ["term.token rewritten after load"], rep.get("doc_changed")
    assert any(p.startswith("the page's script changed 1 popup entr") for p in rep["problems"]), \
        rep["problems"]
    # and a page that is not an explainer page is refused, not passed
    with pytest.raises(ValueError):
        CK.instrument("<html>no body element here</html>")


def test_item_9_the_newest_real_run_page_runs_in_a_headless_browser(tmp_path):
    """The same check on the newest real run under Saved, whose snapshots carry real
    attention maps, eye pictures and every optional block - the data the fixture leaves
    out. Skips where there is no run data (a git worktree has no Saved folder). The page is
    rendered into the test's temp folder: BX.build would write into the run folder."""
    CK, exe = _browser_or_skip()
    run = BX.newest_run()
    if not run or not os.path.isdir(os.path.join(run, "explain")):
        pytest.skip("no run with snapshots under Saved/mimo_bridge on this checkout")
    page = tmp_path / "real.html"
    page.write_text(BX.render(run), encoding="utf-8", newline="")
    assert page.stat().st_size <= BX.PAGE_BUDGET_MB * 1e6
    rep = CK.check_page(str(page), exe)
    assert rep["ran"], rep.get("why")
    assert not rep["problems"], "\n".join(rep["problems"])


def test_item_8_the_landing_token_estimate_is_the_formula_and_is_called_an_estimate(tmp_path):
    """Item 8. A page whose snapshots do not record the size of the token the recalled bias
    lands on (every snapshot taken before 2026-09-19) gets an ESTIMATE of it from the newest
    checkpoint's final LayerNorm, sqrt(mean gain^2 + mean bias^2). It is a formula under an
    assumption, not the measurement, and the label has to keep saying so."""
    import numpy as np
    from safetensors.numpy import save_file
    run = _fixture_run(tmp_path)
    assert BX.landing_token_estimate(run) is None, "no checkpoint file: nothing to estimate from"
    g = np.linspace(0.5, 2.5, 64).astype(np.float32)
    b = np.linspace(-1.0, 1.0, 64).astype(np.float32)
    for step, scale in ((100, 1.0), (250, 2.0)):
        d = os.path.join(run, "checkpoints", f"ckpt_step_{step}")
        os.makedirs(d, exist_ok=True)
        save_file({"forward_model.final_norm.weight": g * scale,
                   "forward_model.final_norm.bias": b * scale}, os.path.join(d, "model.safetensors"))
    est = BX.landing_token_estimate(run)
    g64, b64 = 2.0 * g.astype(np.float64), 2.0 * b.astype(np.float64)
    want = float(np.sqrt((g64 ** 2).mean() + (b64 ** 2).mean()))
    assert est["step"] == 250, "the estimate must come from the NEWEST checkpoint"
    assert abs(est["estimate"] - want) < 1e-3
    assert abs(est["estimate"] ** 2 - est["gain_rms"] ** 2 - est["bias_rms"] ** 2) < 1e-2
    assert _blocks(BX.render(run))["data-meta"]["memory_token_estimate"] == est
    # a checkpoint without that layer gives no estimate, never a guess
    save_file({"something.else": g}, os.path.join(run, "checkpoints", "ckpt_step_250", "model.safetensors"))
    assert BX.landing_token_estimate(run) is None
    js = _viewer_js()
    mem = js[js.index("const MEM_EST ="):js.index("const MEM_FMR =")]
    assert '"an ESTIMATE from the final LayerNorm' in mem and "measured on H81" in mem
    assert "MEM_TOKEN !== null ||" in mem, "the estimate is drawn even when the measurement exists"


def test_the_memory_popups_state_this_runs_own_bias_against_its_token(tmp_path):
    """2026-09-19: both memory popups typed H81's measurement ("a third to a half the size of the token
    it lands on") on every run's page; H84 measures 1.18 times. The sentence is now built from the run's
    newest snapshot, says so when the snapshots do not carry the pair, and always names H81's figure as
    H81's."""
    run = _fixture_run(tmp_path)
    name = os.path.basename(run)
    absent = BX.memory_bias_fill(run, name)["{memory_bias_vs_token}"]
    assert "do not record the size of the token" in absent and "On run H81's probe" in absent
    idx = json.load(open(os.path.join(run, "explain", "index.json")))["snapshots"]
    newest = os.path.join(run, "explain", idx[-1]["file"])
    snap = json.load(open(newest))
    snap["memory"] = {"bias_rms": 4.27972, "bias_rms_body": 4.33954, "landing_token_rms": 3.69122}
    json.dump(snap, open(newest, "w"))
    got = BX.memory_bias_fill(run, name)["{memory_bias_vs_token}"]
    # DESIGN.md section 11 (2026-09-23): the size against the token is a percentage of it
    assert f"{100 * 4.33954 / 3.69122:.0f}% of the size of the token it lands on" in got
    assert f"Measured on {name}'s newest snapshot (step {idx[-1]['step']:,}, body" in got
    assert "On run H81's probe it measured a third to a half" in got
    page = BX.render(run)
    assert "{memory_bias_vs_token}" not in page and "118% of the size of the token" in page
    for key in ("node.memory", "edge.memory.final_norm"):
        texts = " ".join(str(v) for v in BX.SCH.DOC[key].values())
        assert "{memory_bias_vs_token}" in texts and "a third to a half" not in texts, key


def test_the_hierarchical_self_popup_answers_plain_questions_with_the_runs_own_numbers(tmp_path):
    """2026-09-19, operator: the popup must be 'clear as day, organized, no gaps, everything defined,
    everything explained, not verbose', and 'things like moments going together is abstract. How?'
    It is one straight path: the job, why 'hierarchical', what goes in, what comes out; the two
    windows, the wait, the state right now; five numbered steps that say HOW a link is worked out,
    with an example; why it sits outside the model; six warnings. Every term it leans on is a glossary
    entry, a slot is never called by another name, the long measurement list is one click away, and
    the numbers are computed for the run on the page by the arithmetic the class uses."""
    from mimo_bridge.hierarchical_self import HierarchicalSelfDark
    hs = HierarchicalSelfDark(["a"], 8, n_slots=4)
    for k in (1, 8, 64, 640, 6400, 64000):
        assert BX.SCH.hself_first_evidence_windows(k) == hs.first_evidence_tick(k), k
    e = BX.SCH.DOC["node.hself"]
    for field, heads in (("w", ("Its job.", "Why 'hierarchical'.", "What goes in.", "What comes out.")),
                         ("when", ("Every model step.", "Two windows.", "A long wait, on purpose.")),
                         # "Right now" sits AFTER the steps, where slot, belongs and cosine are defined
                         ("h", ("1. **Store kinds of model steps.", "2. **Match.", "3. **Share out the model step.",
                                "4. **Look for a link.", "5. **Rule out luck.", "**Right now.** {hself_now}",
                                "**Why it sits outside the model."))):
        for head in heads:
            assert head in e[field], (field, head)
    # HOW a link is worked out, not only that one exists
    for phrase in ("two running averages", "UP when the quick one is above", "gives a plus",
                   "gives a minus", "Example:"):
        assert phrase in e["h"], phrase
    said = " ".join(e.values())
    assert len(said.split()) < 830, "the popup has grown past 830 words: %d" % len(said.split())
    assert e["h"].index("2. **Match.") < e["h"].index("**Right now.**"), "terms used before they are defined"
    for term in ("slot", "presence", "wellness-link", "evidence-window", "sign-flipped-control", "cosine",
                 "hself-measurements"):
        assert f"[[{term}" in said, f"the popup leans on '{term}' without linking its definition"
        assert "term." + term in BX.SCH.DOC, term
    for other_name in ("table row", "situation table", "situation slot"):
        assert other_name not in said, f"a slot is called '{other_name}'"
    args = {"hself_slots": 6400, "caregiver_mode": "responsive", "caregiver_present_seconds": 60,
            "caregiver_absent_seconds": 15}
    f = BX.SCH.hself_fill(args, 0.05, "H99")
    assert f["{hself_slots}"] == "6,400" and f["{hself_l2_fast_s}"] == "50"
    assert f["{hself_l2_window_min}"] == "8.3" and f["{hself_l3_window_min}"] == "83"
    first = f["{hself_first_evidence_text}"]
    assert "needs 27 windows before it counts (2 for the luck test to warm up, 25 for a perfect score" in first
    assert "270,000 model steps for Level 2 (3.8 hours of body time)" in first
    assert "2,700,000 for Level 3 (37.5 hours)" in first
    assert "once every 75 s" in f["{hself_window_anchor}"] and "about 7 visits" in f["{hself_window_anchor}"]
    assert "about 67" in f["{hself_window_anchor}"]
    assert "needs 16 windows" in BX.SCH.hself_fill({"hself_slots": 64}, 0.05, "H99")["{hself_first_evidence_text}"]
    assert BX.SCH.hself_fill({"hself_slots": 64}, 0.05, "H99")["{hself_window_anchor}"] == ""
    never = BX.SCH.hself_fill({"hself_slots": 64000}, 0.05, "H99")
    assert "nothing can ever count" in never["{hself_first_evidence_text}"]
    assert BX.hself_wall_fill([{"model_steps_per_s": 1.1}], never, "H99")["{hself_first_evidence_wall}"] == ""
    wall = BX.hself_wall_fill([{"model_steps_per_s": 1.1}] * 5, f, "H99")["{hself_first_evidence_wall}"]
    assert "1.10 model steps per second" in wall and "about 68 hours" in wall and "about 28 days" in wall
    assert BX.hself_wall_fill([], f, "H99")["{hself_first_evidence_wall}"] == ""
    # the state right now comes from the run's newest snapshot, under either name of the two gauges
    run = _fixture_run(tmp_path)
    assert BX.hself_now_fill(run, "H99") == {}
    idx = json.load(open(os.path.join(run, "explain", "index.json")))["snapshots"]
    newest = os.path.join(run, "explain", idx[-1]["file"])
    snap = json.load(open(newest))
    snap["hself"] = {"slots_filled": 6400, "moments_with_a_slot": 0.0, "best_slot_cos": 0.1621,
                     "l2_ticks": 11, "l2_ticks_needed": 25, "l2_sig": -1,
                     "l3_ticks": 0, "l3_ticks_needed": 25, "l3_sig": -1}
    json.dump(snap, open(newest, "w"))
    now = BX.hself_now_fill(run, "H99")["{hself_now}"]
    assert "6,400 slots in use" in now and "0 of 100 model steps belong to a slot" in now
    # DESIGN.md section 11: the likeness is placed against the match rule's line, never printed
    assert "nearest slot sits below the nine tenths a match needs" in now and "every score is about old model steps" in now
    assert "Level 2 has lived 11 of the 25 windows it needs and Level 3 has lived 0 of the 25" in now
    assert now.endswith("Neither can count yet.")
    snap["hself"]["moments_with_a_row"] = snap["hself"].pop("moments_with_a_slot")
    snap["hself"]["best_row_cos"] = snap["hself"].pop("best_slot_cos")
    json.dump(snap, open(newest, "w"))
    assert "0 of 100 model steps belong to a slot" in BX.hself_now_fill(run, "H99")["{hself_now}"]
    page = BX.render(run)
    assert "{hself_" not in page and "0 of 100 model steps belong to a slot" in page


def test_the_hierarchical_self_popup_says_when_the_table_is_switched_off(tmp_path):
    """H84 has run with --hierarchical-self off since segment 9 (2026-09-20 05:01) and the
    named-things tracker in its place, but the popup still described a table that stores
    patterns every moment, the panel was still titled "watching only", and the filled line
    read "the snapshots do not record it" - which is false: capture.py writes
    "absent: hierarchical self off" into every snapshot of such a run."""
    on = BX.SCH.hself_fill({"hself_slots": 6400, "hierarchical_self": "dark",
                            "entity_self": "off"}, 0.05, "RUNON")
    assert on["{hself_off_notice}"] == "", "a running table must not carry the off notice"
    off = BX.SCH.hself_fill({"hself_slots": 6400, "hierarchical_self": "off",
                             "entity_self": "dark"}, 0.05, "RUNOFF")
    notice = off["{hself_off_notice}"]
    assert notice.startswith("**Switched off in RUNOFF.**"), notice
    assert "--hierarchical-self off" in notice and "named-things panel" in notice
    assert "It is switched off in RUNOFF" in off["{hself_now}"]
    assert "{hself_off_notice}" in BX.SCH.DOC["node.hself"]["w"]
    # the panel title, so a reader of the page sees it without opening the popup
    js = _viewer_js()
    assert "Hierarchical self (switched off on this run)" in js
    assert "Hierarchical self (dark - watching only)" in js


def test_the_filled_line_reads_a_switched_off_snapshot_rather_than_claiming_silence(tmp_path):
    """hself_now_fill returned nothing for a run whose snapshots carry the string
    "absent: hierarchical self off", so the page fell back to "RUN's snapshots do not
    record it"."""
    run = _fixture_run(tmp_path)
    for name in sorted(os.listdir(os.path.join(run, "explain"))):
        if not name.startswith("step_"):
            continue
        path = os.path.join(run, "explain", name)
        snap = json.load(open(path, encoding="utf-8"))
        snap["hself"] = "absent: hierarchical self off"
        json.dump(snap, open(path, "w", encoding="utf-8"))
    fill = BX.hself_now_fill(run, "RUNOFF")
    assert "RUNOFF's newest snapshot records the table as switched off" in fill["{hself_now}"]
    assert "do not record it" not in fill["{hself_now}"]


# ======================================================================================
# the six page/code disagreements measured on 2026-09-21: how many wellness formulas score
# a moment, which reading of the body map the coherence term carries, object incorporation
# (UEMA 11.2), the hours level of the named-entity self, the push columns of a touch token,
# and the touch row's shape
# ======================================================================================
def _all_doc_text(doc):
    return " ".join(str(v) for entry in doc.values() for v in entry.values())


def test_the_page_counts_the_wellness_formulas_the_code_runs():
    """The page said FIVE fixed formulas and drew a box labelled "5 grounded + approximate".
    A lived moment is scored by SIX: the five in lem/model/wellness.py plus self_touch,
    which lem_model.py adds to the same set of scores. The subtle half is that the TRAINED
    copy - the one that scores an IMAGINED moment for the planner - has one output per name
    in lem/model/wellness.py APPROXIMATE_TERMS, and self_touch is not among them, so the
    planner cannot see that term when it scores an imagined moment."""
    import inspect
    from lem.model.wellness import APPROXIMATE_TERMS, WellnessEvaluators
    from lem.training import loss_manager as LM
    from mimo_bridge.lem_model import LemLite
    # the code's own counts, so a term added or removed fails here before the page lies
    assert len(WellnessEvaluators.EVALUATOR_NAMES) + 1 == 6      # + self_touch
    assert len(APPROXIMATE_TERMS) == 5 and "self_touch" not in APPROXIMATE_TERMS
    # the lesson that trains the copy reads that same list, so "copies five" is a statement
    # about the lesson as well as about the network's width
    assert "APPROXIMATE_TERMS" in inspect.getsource(LM.LossManager._compute_wellness_loss)
    step = inspect.getsource(LemLite.step)
    assert "wellness_out['grounded']['self_touch']" in step, \
        "the sixth term no longer joins the grounded set the page describes"

    doc = _page_doc()
    w = doc["term.wellness"]
    assert "six fixed formulas" in w["w"]
    assert "[[eval-self-touch|self-contact increase]]" in w["w"]
    assert "APPROXIMATE_TERMS" in w["h"]
    assert "first five scores" in w["w"] and "no self-touch output" in w["w"]
    # THE NEW TRUTH (2026-09-21 in the code, 2026-09-22 on the page): the term pays own parts
    # ARRIVING - the rise since the last moment of column 12's count - not the share of felt
    # contact that is the body's own, which is what this entry and its pin used to say
    st = doc["term.eval-self-touch"]
    stw = inspect.getsource(LemLite._self_touch_wellness)
    assert "tactile[..., 12]" in stw and "(parts - prev).clamp_min(0.0)" in stw, (
        "the term no longer pays the rise of column 12: rewrite the entry")
    assert "Subtract the previous model step" in st["w"] and "discard negative differences" in st["w"]
    assert "column 12" in st["h"] and "_self_touch_wellness" in st["h"]
    assert "every contact it feels is another part of itself" not in st["w"]
    # the gate's own numbers, from the constants that set them
    from mimo_bridge.lem_model import SELF_TOUCH_READINGS, SELF_TOUCH_STALE_STEPS
    assert f"last {SELF_TOUCH_READINGS} controlled self-touch readings" in st["f"]
    assert f"{SELF_TOUCH_STALE_STEPS:,} model steps old" in st["f"]
    assert "exceed twice its standard error" in st["f"]
    gate = inspect.getsource(LemLite._self_touch_gate)
    assert "mean > 2.0 * err" in gate, "the gate's line is no longer twice the standard error"
    assert "len(ring) < SELF_TOUCH_READINGS" in gate and "age > SELF_TOUCH_STALE_STEPS" in gate
    assert "self-contact increase" in doc["node.wellness"]["w"]
    assert "Fixed scores are calculated each model step" in doc["node.wellness"]["when"]
    assert "Only the motor-state position formula" in doc["edge.wellness.motor"]["w"]
    assert "fixed-formula outputs" in doc["drives.wellness"]["w"]
    assert "six fixed formulas" in doc["node.hself"]["w"]
    # no page text may still count five of them
    joined = _all_doc_text(doc).lower()
    for stale in ("five fixed formulas", "5-term", "five [[wellness]] evaluators",
                  "five grounded", "five evaluators' sum", "5 grounded"):
        assert stale not in joined, f"page text still counts the wellness terms as {stale!r}"
    js = _viewer_js()
    assert '"5 grounded + approximate"' not in js
    assert '"6 fixed + a learned copy of 5"' in js
    assert "the planner's copy of wellness has no output for it" in js
    # the flag has to reach the page, or the run's own weight cannot be printed
    assert "self_touch_wellness" in BX.PUBLISHED_ARGS


def test_the_wellness_weight_sentence_is_the_runs_own(tmp_path):
    """{self_touch_weight} names the run's own --self-touch-wellness, and a run that did
    not record the flag says so rather than printing another run's setting."""
    zero = BX.SCH.wellness_fill({"self_touch_wellness": 0.0}, "RUNZERO")["{self_touch_weight}"]
    assert "RUNZERO was launched with --self-touch-wellness 0" in zero
    assert "reads exactly zero on every step" in zero
    paid = BX.SCH.wellness_fill({"self_touch_wellness": 0.25}, "RUNPAID")["{self_touch_weight}"]
    assert "--self-touch-wellness 0.25" in paid and "adds 0.25" in paid
    none = BX.SCH.wellness_fill({}, "RUNNONE")["{self_touch_weight}"]
    assert "do not record --self-touch-wellness" in none
    pick = BX.SCH.wellness_fill({"wellness_coherence": "certainty"}, "RUNC")
    assert "--wellness-coherence certainty" in pick["{wellness_coherence_choice}"]
    assert "rises as the handles' co-movement numbers fall" in pick["{wellness_coherence_choice}"]
    run = _fixture_run(tmp_path)
    args_path = os.path.join(run, "args.json")
    args = json.load(open(args_path))
    args["self_touch_wellness"], args["wellness_coherence"] = 0.25, "extent"
    json.dump(args, open(args_path, "w"))
    page = BX.render(run)
    assert "{self_touch_weight}" not in page and "{wellness_coherence_choice}" not in page
    assert "--self-touch-wellness 0.25" in page


def test_the_coherence_entry_describes_the_reading_the_runs_carry():
    """The page said this slot carries the body map's CERTAINTY and cited coherence_score().
    The runs carry EXTENT: args.json records wellness_coherence "extent" - the launcher's
    default - and lem_model._schema_coherence then picks self_extent(). Certainty was
    retired because it equalled 1 - 2 x the handles' mean number, so it rose as that number fell.
    Since 2026-10-03 (the operator's decision) the entry names what the statistic is: co-movement
    of command changes and proprioception changes in the same model step, not causation, with its
    input, its computation and its output."""
    import inspect
    from mimo_bridge import train_mimo as TM
    from mimo_bridge.lem_model import LemLite
    src = inspect.getsource(LemLite._schema_coherence)
    assert 'self.wellness_coherence == "extent"' in src and "sc.self_extent()" in src
    assert "sc.coherence_score()" in src, "the other reading is no longer reachable"
    parser = " ".join(inspect.getsource(TM).split())
    assert '"--wellness-coherence", choices=["extent", "certainty"], default="extent"' in parser

    c = _page_doc()["term.eval-coherence"]
    assert "co-movement" in c["w"] and "does not establish that the command caused" in c["w"]
    # Mechanism and configuration remain pinned; historical fixture measurements are no longer prose.
    assert "command changed" in c["w"] and "proprioception measurement changed" in c["w"]
    assert "running root-mean-square" in c["h"] and "sigmoid" in c["h"]
    assert "extent setting" in c["w"] and "handles and tracked outside objects" in c["w"]
    assert "self_extent" in c["h"] and "_schema_coherence" in c["h"]
    assert "7.6%" in c["h"] and "92.4%" in c["h"]
    assert "certainty alternative" in c["f"] and "cosine similarity" in c["f"]
    assert "October 3 operator decision" in c["f"]
    assert "coherence_source" in _viewer_js(), \
        "the viewer never says which reading a snapshot's coherence score carried"


def test_the_object_incorporation_panel_states_its_scale_its_control_and_what_the_world_allows():
    """UEMA 11.2 was absent from the page: incorporation_stats() went into every stats
    record and neither the snapshot nor the viewer carried it. The panel must give, per
    thing, what it scored, the same score with the hand moving the opposite way, the
    difference, and whether the world lets the body move that thing at all - the toy is a
    MuJoCo mocap body, so physics never moves it and a toy score can only ever be the adult
    carrying it the way a hand happened to go."""
    import inspect
    import math
    from lem.model.body_schema import DifferentiableBodySchema
    from mimo_bridge.lem_model import LemLite, object_physics_driven_for
    stats_src = inspect.getsource(LemLite.incorporation_stats)
    for field in ("handles_mine", "extent", "certainty", "carried_by_wellness", "objects",
                  "objects_per_body", "objects_evidence", "objects_physics_driven"):
        assert field in stats_src, f"incorporation_stats() no longer writes {field}"
    # the world's own report: a mocap toy cannot be displaced by the body
    assert object_physics_driven_for({"caregiver": {}})["toy"] is False
    t = DifferentiableBodySchema(n_handles=4, object_names=("toy",)).temperature
    floor, top = 1 / (1 + math.exp(t * 0.5)), 1 / (1 + math.exp(-t * 0.5))
    assert round(floor, 4) == 0.0759 and round(top, 4) == 0.9241
    # certainty is mean(|number - 0.5| x 2) over numbers that live on that scale, so its
    # own top is not 1 either
    assert round(2 * (top - 0.5), 4) == 0.8483

    d = _page_doc()["incorporation"]
    # DESIGN.md section 11 (2026-09-23): the two ends are percentages of the 0-to-1 range, and
    # every number is written as the percentage of the way between them
    assert f"{100 * floor:.1f}% of the 0-to-1 range" in d["w"] and f"{100 * top:.1f}%" in d["w"], \
        "the two ends of the scale are not stated"
    assert "0% = a score of 0" in d["w"] and "100% = a score of 1" in d["w"]
    assert "co-movement" in d["t"] and "It does not say the command caused the change" in d["w"]
    assert "0% when every number sits exactly on the middle to 100% when every number is at one end" in d["w"], \
        "the certainty row's own ends are not stated"
    assert "sigmoid(5 x (score - one half))" in d["w"]
    assert "the same score with that handle's direction reversed" in d["w"]
    assert "mocap" in d["w"] and "never integrates" in d["w"]
    assert "incorporation_stats()" in d["h"] and "object_evidence()" in d["h"]
    js = _viewer_js()
    assert '"incorporation"' in js, "no panel opens the incorporation popup"
    assert "function incorporationTable(" in js
    for field in ("objects_evidence", "objects_physics_driven", "carried_by_wellness",
                  "handles_mine", "objects_per_body"):
        assert field in js, f"the panel never reads {field}"
    assert "INCORP_FLOOR" in js and "INCORP_TOP" in js
    assert "0.0759" not in js and "0.9241" not in js, \
        "the scale's two ends must be computed from the map's own temperature, not typed"
    cap = open(os.path.join(BX.PKG, "capture.py"), encoding="utf-8").read()
    assert '"incorporation": incorporation_blk' in cap
    assert "model.incorporation_stats()" in cap


def test_the_named_things_panel_draws_both_levels_with_their_counts():
    """The panel's blurb promised "the same two levels" and said a count of -1 means
    "cannot be measured yet", while the table drew the minutes level only and no count at
    all. Every per-level field EntitySelfDark.stats() writes must be read by the table."""
    import inspect
    from mimo_bridge.entity_self import EntitySelfDark
    stats_src = inspect.getsource(EntitySelfDark.stats)
    js = _viewer_js()
    table = js[js.index("function entitySelfTable("):js.index("function hselfGroups()")]
    for suffix in ("_ticks", "_window_moments", "_entity_windows", "_link", "_rank",
                   "_line", "_counted", "_counted_names", "_steps_to_first_evidence"):
        assert 'f"{tag}%s"' % suffix in stats_src, f"stats() no longer writes {suffix}"
        assert '"%s"' % suffix in table, \
            f"the table never reads {suffix} - written for both levels, drawn for neither"
    assert '[["l2", "minutes"], ["l3", "hours"]]' in js, "the table still draws one level"
    assert "cannot be measured yet" in table
    assert "named things with a proven wellness link" in table
    assert "model steps to go before anything here could count" in table
    assert "best standing against its own 8 controls" in table
    blurb = js[js.index('"Named things the self-model watches (dark)"'):
               js.index("entitySelfTable(snap.entity_self)")]
    assert "minutes and hours" in blurb, "the blurb promises two levels without naming them"
    assert "a count of -1 means 'cannot be measured yet'" in blurb


def test_the_touch_panel_draws_the_push_columns_and_reads_the_legend_the_capture_writes():
    """Columns 13 to 17 of a touch token - the push's direction, its depth and its arrival
    speed - ride every row, and the panel drew none of them; senses.touch_cloud_doc, the
    column legend the capture writes, was read by no viewer code. That legend is now what
    tells the page which column is which, so the two cannot drift apart."""
    import numpy as np
    import torch
    from mimo_bridge.explainer import capture as CAP
    from mimo_bridge.lem_model import TOUCH_TOKEN_F, TOUCH_VALID_COL, touch_point_cloud

    def legend_columns(text):
        """viewer.js touchLegend, mirrored: each comma-separated entry is one column,
        except an entry ending in xN, which is N."""
        out, col = [], 0
        for raw in text.strip().lstrip("[").rstrip("]").split(","):
            label = raw.strip()
            if not label:
                continue
            m = re.search(r"\bx(\d+)$", label)
            span = int(m.group(1)) if m else 1
            out.append((label, col, span))
            col += span
        return out, col

    entries, total = legend_columns(CAP.TOUCH_CLOUD_DOC)
    assert total == TOUCH_TOKEN_F, \
        f"the legend accounts for {total} columns and a touch row carries {TOUCH_TOKEN_F}"
    at = {label: start for label, start, _ in entries}
    assert at["valid"] == TOUCH_VALID_COL
    # the columns the panel draws, checked against a token the code itself builds
    imp = torch.zeros(256, 10)
    imp[0, 0] = 0.9                                     # one part pressed now
    self_cols = np.zeros((256, 2), dtype=np.float32)
    self_cols[0] = (0.75, 0.5)                          # self-touch share, own parts / 4
    contact = np.zeros((256, 5), dtype=np.float32)
    contact[0] = (0.1, 0.2, 0.3, 0.4, 0.5)              # push x/y/z, depth, arrival speed
    row = touch_point_cloud(imp, None, self_cols, contact)[0]
    assert float(row[at["pressure"]]) == pytest.approx(0.9)
    assert float(row[at["self-touch share"]]) == pytest.approx(0.75)
    assert [float(row[at["push direction x3"] + i]) for i in range(3)] == \
        pytest.approx([0.1, 0.2, 0.3])
    depth_label = next(lb for lb in at if lb.startswith("depth"))
    arrival_label = next(lb for lb in at if lb.startswith("arrival speed"))
    assert float(row[at[depth_label]]) == pytest.approx(0.4)
    assert float(row[at[arrival_label]]) == pytest.approx(0.5)
    # the legend names the divisors in force, which is what the viewer reads them from
    assert depth_label.endswith("/%g" % PUSH_DEPTH_SCALE_CM), depth_label
    assert arrival_label.endswith("/%g" % PUSH_ARRIVAL_SCALE_CMS), arrival_label
    assert float(row[TOUCH_VALID_COL]) == 1.0

    js = _viewer_js()
    touch = js[js.index("// --- touch"):js.index("// --- attention")]
    assert "touchLegend(snap.senses.touch_cloud_doc)" in touch, "the legend is still unread"
    assert "which way the push came from" in touch and "how fast it arrived" in touch
    # the unit rides every cell (DESIGN.md section 11: a number a person reads carries its unit)
    assert 'fmt(deep, 2) + " cm"' in touch and 'fmt(spd, 1) + " cm/s"' in touch
    # the fallbacks typed for a snapshot with no legend must be the code's own columns
    for label, pattern in (("pressure", r"touchCol\(lg, /\^pressure\$/, (\d+)\)"),
                           ("self-touch share", r"touchCol\(lg, /self-touch share/, (\d+)\)"),
                           ("push direction x3", r"touchCol\(lg, /push direction/, (\d+)\)"),
                           (depth_label, r"touchCol\(lg, /\^depth/, (\d+)\)"),
                           (arrival_label, r"touchCol\(lg, /\^arrival speed/, (\d+)\)")):
        m = re.search(pattern, touch)
        assert m, f"the panel no longer reads {label}"
        assert int(m.group(1)) == at[label], \
            f"the fallback column for {label} is {m.group(1)}; the code writes it at {at[label]}"
    # the divisors are read from each snapshot's OWN legend, so a printed centimetre is the
    # legend's centimetre whichever scale the snapshot was written on. The fallback, for a
    # snapshot carrying no legend at all, stays at 10 and 500: every snapshot without a legend
    # was written before the push's scales changed on 2026-09-21.
    assert re.search(r"touchScale\(lg, /\^depth/, 10\)", touch)
    assert re.search(r"touchScale\(lg, /\^arrival speed/, 500\)", touch)


def test_the_touch_row_shape_on_the_page_is_the_codes_own_constants():
    """The guard the page never had: the text says "Up to 32 rows x 19 numbers" and "Each
    row is 19 numbers", and nothing tied either figure to lem_model.TOUCH_TOKENS and
    TOUCH_TOKEN_F. The row has been widened twice (12 -> 14 -> 19) and the box still said
    12 on 2026-09-21, seven columns behind the code."""
    import torch
    from mimo_bridge.lem_model import (TOUCH_TOKENS, TOUCH_TOKEN_F, TOUCH_VALID_COL,
                                       touch_point_cloud)
    assert TOUCH_VALID_COL == TOUCH_TOKEN_F - 1
    assert tuple(touch_point_cloud(torch.zeros(256, 10), None).shape) == (TOUCH_TOKENS,
                                                                          TOUCH_TOKEN_F)
    doc = _page_doc()
    edge = doc["edge.sense_tactile.enc"]["w"]
    assert f"Up to {TOUCH_TOKENS} rows x {TOUCH_TOKEN_F} numbers" in edge, \
        "the data-contract popup's row shape is not this build's touch token"
    sense = doc["node.sense.tactile"]
    assert f"Each row is {TOUCH_TOKEN_F} numbers" in sense["w"], \
        "the touch box states a row width that is not this build's"
    assert f"cut at {TOUCH_TOKENS}" in sense["w"] and f"keeps the first {TOUCH_TOKENS}" in sense["h"]
    joined = _all_doc_text(doc)
    for stale in ("Each row is 12 numbers", "Each row is 14 numbers",
                  "32 rows x 12 numbers", "32 rows x 14 numbers"):
        assert stale not in joined, f"page text still says {stale!r}"


# ======================================================================================
# the gain-of-function build (2026-09-22): the page documents every feature the build added,
# the stale items the explainer survey listed are fixed, and a page built from a run that has
# every feature on draws each one - checked in Python and in a headless browser
# ======================================================================================
_GOF_SENSES = ["vision", "tactile", "audio", "proprioception", "motor", "vestibular"]
_GOF_CHANNELS = ["integrity", "controllability", "learning_progress", "coherence",
                 "contact_control", "consequence", "agency", "prediction_error",
                 "tactile_comfort", "capability", "movement", "curiosity", "efficiency"]


def _gof_snapshot_blocks(step):
    """The schema-v3 blocks as capture.py writes them (shapes from its own functions)."""
    from mimo_bridge.vestibular_adapter import VESTIBULAR_FIELDS
    alts = ["hold", "reflex", "perturb_1", "perturb_2"]
    from mimo_bridge.explainer.capture import _vestibular_units
    units = _vestibular_units()            # capture.py's own lived unit per field
    fields = [{"name": n, "stored": 0.1 * (i % 5), "lived": 0.1 * (i % 5), "unit": units[n][1]}
              for i, n in enumerate(VESTIBULAR_FIELDS)]
    fields[-1].update(stored=1.0, lived=1.0)
    return {
        "causal_chain": {
            "step": step, "body": 0, "horizon": 2, "route": "acting", "alternatives": alts,
            "goal": {"ring_index": 12, "age_steps": 187, "ring_size": 200, "value_ranked": 1.42,
                     "value_stored": 4.9, "bodily": 3.9, "consequence": 0.5,
                     "curiosity_factor": 1.0, "other_drives": 0.5, "locality": False},
            "plan0": {"pos_step_cm": 7.4, "strength_mean": 0.47, "gaze_az_deg": 3.2,
                      "gaze_el_deg": -1.5},
            "pred_change": {m: 0.02 for m in _GOF_SENSES if m != "motor"},
            "pred_wellness": dict({"chosen": 3.91}, **{a: 3.9 + 0.01 * i for i, a in enumerate(alts)}),
            "goal_distance": dict({"chosen": 1.2}, **{a: 1.3 for a in alts}),
            "chosen_rank": 3, "pred_wellness_terms_chosen": [0.98, 0.5, 0.8, 0.3, 0.96],
            "terms": ["prediction_error", "tactile_comfort", "capability", "coherence", "motor_state"],
            "wellness_rank": {"contrib_chosen": {c: (0.3 if i % 2 else -0.2) for i, c in enumerate(_GOF_CHANNELS)},
                              "lead_over_runner_up": {c: 0.05 for c in _GOF_CHANNELS},
                              "gated_share": 0.0, "all_gated": False,
                              "channels_now": {c: 0.5 for c in _GOF_CHANNELS},
                              "planner_blind": ["controllability", "learning_progress"]},
            "reached_body": True, "order_seq": 41, "plan_share_of_executed": 0.35},
        "efference": {
            "handle_names": ["wrist_L", "gaze", "head"],
            "tenure": {"step": step, "order_began": step - 5,
                       "expected_cm": [[0.5, 0.1, 0.0], [0, 0, 0], [0.2, 0.2, 0.1]],
                       "expected_deg": [[3.0, 1.0, 0.0], [0, 0, 0], [2.0, 0.5, 0.1]],
                       "achieved_cm": [[1.5, 0.3, 0.2], [0, 0, 0], [0.4, 0.1, 0.0]],
                       "achieved_deg": [[9.0, 2.0, 1.0], [0, 0, 0], [5.0, 0.5, 0.2]]},
            "window_so_far": {"body_cm_err_next": 1.3, "body_cm_do_nothing_next": 1.32,
                              "body_cm_shuffled_next": 1.31, "gaze_deg_err_1": 4.5,
                              "gaze_deg_do_nothing_1": 4.1},
            "scales": {"pos_next_cm": 0.452}, "weight": 0.1},
        "vestibular": {"fields": fields, "read_by_model": True, "valid": True},
        "wellness_channels": {
            "mode": "normalized", "curiosity_mode": "additive", "consequence_world_only": True,
            "gate_line": 0.87, "pain": 0.77,
            "channels": [{"channel": c, "value": 0.5, "centre": 0.45, "scale": 0.02,
                          "floor": 0.01, "weight": 0.5,
                          "weight_in_force": 0.0 if c == "controllability" else 0.5,
                          "contribution": 1.25 if i % 3 else -0.75, "planner_sees": "no"}
                         for i, c in enumerate(_GOF_CHANNELS)],
            "controllability": {"value": 0.02, "gate_open": False, "gate_mean": 0.001,
                                "gate_line": 0.02, "gate_why": "under the line", "last": None},
            "learning_progress": {"channel": 0.01, "per_sense": {
                "vision": {"progress": 0.03, "ratio_fast": 1.6, "ratio_slow": 1.7},
                "tactile": {"progress": -0.01, "ratio_fast": 1.1, "ratio_slow": 1.09}}}},
        "contact_control": {"step": step, "valid": True, "control_moment": False, "gate_open": False,
                            "self_predicted": [["left_hand", "left_thdistal", 1.0]],
                            "world_predicted": [["right_foot", 0.5]], "would_pay": 2.5, "pay": 0.0,
                            "channel_weight": 0.5, "gate_mean": None, "gate_line": None,
                            "gate_why": "12 of the 32 controlled readings needed",
                            "margin": {"self": 0.114, "world": 0.176}},
        "synergy": {"dim": 16, "babble_sigma": 0.8, "stamp": {"dim": 16}},
    }


def _gof_record(i, step):
    ch = {c: 1.0 / len(_GOF_CHANNELS) for c in _GOF_CHANNELS}
    per = {m: 0.01 + 0.001 * i for m in _GOF_SENSES if m != "motor"}
    return {
        # the collapse watch on every body (every report since 2026-09-25; chart 7b)
        "token_temporal_std_bodies": {m: [0.4, 0.1 + 0.001 * i] for m in _GOF_SENSES},
        "causal_chain": {"events": 3, "planner_events": 10, "latch_predictions": 20,
                         "chosen_best_share": 0.2 + 0.01 * i, "alternatives_n": 4,
                         "chosen_closest_share": 0.6, "reached_body_share": 0.5,
                         "pred_err_next": per, "persist_err_next": {m: 0.02 for m in per},
                         "pred_err_latch": per, "persist_err_latch": {m: 0.03 for m in per},
                         "predicted_wellness_change": 0.001, "realized_wellness_change": 0.002,
                         "wellness_change_corr": 0.1,
                         # the field contract of 2026-09-23: the tie band and the plan-to-sent distance
                         "chosen_tied_share": 0.3, "tie_band": 0.016, "gap_to_best_alt_mean": 0.002,
                         "candidate_spread_mean": 0.005, "plan_exec_distance_cm": 1.5},
        "efference": {"body_cm_err_next": 1.3, "body_cm_do_nothing_next": 1.32,
                      "body_cm_shuffled_next": 1.31, "body_cm_err_latch": 5.2,
                      "body_cm_do_nothing_latch": 5.3, "body_cm_shuffled_latch": 5.25,
                      "gaze_deg_err": [4.5, 6.0, 8.0], "gaze_deg_do_nothing": [4.1, 6.2, 8.8],
                      "gaze_deg_shuffled": [4.5, 6.0, 8.0],
                      # the size-matched control (2026-10-02, lem/model/efference.py window_summary)
                      "body_cm_size_matched_next": 1.305, "body_cm_size_matched_latch": 5.22,
                      "gaze_deg_size_matched": [4.45, 6.05, 8.1]},
        "vestibular": {"pred_err": 0.04, "persist_err": 0.02, "head_tilt_deg": 30.0,
                       "trunk_roll_deg": 20.0, "trunk_pitch_deg": 3.0},
        "controllability": {"agreement_mean": 0.02, "control_mean": 0.0, "measured_share": 0.9,
                            "difference_mean": 0.02, "gate_open": False,
                            "gate_why": "difference 0.0200 under 2 x SE 0.0310",
                            # the gate in force beside the recorded-only shadow gate (2026-10-03,
                            # lem/model/wellness_channels.py ControllabilityTracker.summary)
                            "size_matched_control_mean": 0.01, "size_matched_difference_mean": 0.01,
                            "shadow_gate_open": False, "gate_commands": 20, "gate_open_commands": 5 + i % 3,
                            "shadow_gate_commands": 20, "shadow_gate_open_commands": i % 3,
                            "gates_same_commands": 20, "gates_both_open_commands": i % 3,
                            "gate_only_open_commands": 5, "shadow_gate_only_open_commands": 0},
        "learning_progress": {"value_mean": 0.01, "per_sense": {"vision": 0.03, "tactile": -0.01},
                              "control_mean": -0.02},
        "wellness_rank": {"winner_margin_share": ch, "integrity_gated_share": 0.0,
                          "all_gated_share": 0.0,
                          "goal_age": {"mean_share_back": 0.55, "oldest_tenth_share": 0.12,
                                       "newest_tenth_share": 0.09, "events": 20},
                          "weight_share_by_channel": {c: 1.0 / len(_GOF_CHANNELS) for c in _GOF_CHANNELS}},
        "contact_control": {"head_auc_bits": {"any": 0.8}, "control_auc_bits": {"any": 0.9},
                            "success_self": 0.1, "success_control": 0.1, "success_world": 0.2,
                            "success_paired": 0.3, "success_paired_world": 0.25,
                            "success_control_world": 0.2,
                            "paid_mean": 0.0, "would_pay_mean": 0.03,
                            "onset_auc_self": 0.61, "onset_auc_world": 0.57,
                            "control_onset_auc_self": 0.55, "control_onset_auc_world": 0.52,
                            "gate_self_open": False, "gate_world_open": False,
                            "habituation_mean": 0.8},
        "self_contact_events": {"onsets_per_body_step": 0.68, "structural_share": 0.125,
                                "in_view_share": 0.008},
        "self_contact_pairs": 1.5, "self_contact_patches": 3.0, "self_contact_force_share": 0.06,
        "hand_press": {"world_share": 0.12, "self_share": 0.66, "world_kpa": 3.0, "self_kpa": 50.0},
        "machine": {"commit_gb": 40.0, "commit_limit_gb": 68.8, "ram_available_gb": 10.0,
                    "rss_gb": 6.0, "step_ms": 800.0, "model_ms": 500.0, "env_ms": 250.0,
                    "causal_ms": 50.0},
        "synergy": {"explained_share": 0.3, "explained_share_unweighted": 0.25, "following_mean": 0.4},
        "next_frame_norm": {"weight_effective": {m: 1.0 for m in per}},
        "prediction_audit": {"modalities": {m: {"improvement_over_persistence": -0.03,
                                                "sensitivity": 0.001,
                                                # the swap test's size-matched arm (2026-10-02)
                                                "improvement_over_size_matched": 0.002 + 0.0001 * i}
                                            for m in per}},
        "prediction_audit_hold_first": {"senses": {m: {"all_bodies": {
            "improvement_over_size_matched": 0.003 + 0.0001 * i}} for m in per}},
        "consequence": {"rate": 0.8, "goal_rate": 1.0, "rate_count_rule": 0.9},
        "incorporation": {"handles_mine": 0.25, "extent": 0.25, "objects": {"toy": 0.3}},
        "entity_self": {"l2_counted": -1, "l3_counted": -1},
        "wellness_align_err": 0.015, "roll_deg_mean": 100.0, "roll_deg_max": 180.0,
        "rolled_over_fraction": 0.75,
        "skin_regions": {"head": {"force_share": 0.2}, "trunk": {"force_share": 0.3}},
        "task_hand_seen": 0.1, "task_mobile_hit": 0.02, "cmd_mag_rad": 0.45,
        # the whole-model liveness block of the field contract (DESIGN.md section 10)
        "liveness": {"layers": {"tenure_heads.pose.0": 0.0, "wellness.approximate.net.0": 0.05,
                                "efference.body.0": 0.9},
                     "pools": {"motor_pool": 0.001, "tactile_pool": 0.09},
                     "line": -3, "samples": 24, "revived": []},
    }


def _fixture_run_gof(tmp_path, monkeypatch):
    """_fixture_run with every feature of the build on: its launch flags, its receipt stamps,
    its stats blocks, its snapshot blocks, a causal_chain.jsonl and a probe index."""
    run = _fixture_run(tmp_path)
    _set_args(run, causal_chain="on", efference_heads="on", efference_weight=0.1,
              vestibular_sense="on", wellness_rank="normalized", controllability="on",
              learning_progress="on", curiosity_mode="additive", curiosity_goal="on",
              consequence_self_contact="exclude", contact_head="on",
              contact_control_wellness=0.5, entity_self="dark", entity_self_teacher="wellness",
              toy_hand_off="follow", contact_profile="mechanical_v4", finger_armature=0.0004,
              intention="on", intention_every=10, smooth_source="executed",
              caregiver_mode="responsive", caregiver_present_seconds=60,
              caregiver_absent_seconds=15)

    def rec_edit(r):
        r["active_modalities"] = list(_GOF_SENSES)
        r["encoders_built"]["vestibular"] = r["encoders_built"].pop("intermind")
        cp = r["component_params"]
        cp["enc_vestibular"] = cp.pop("enc_intermind")
        cp.update(efference_heads=500000, contact_head=90000, motor_synergy=220000)
        r["total_params"] = sum(cp.values())
        r["checkpoint_inventory"]["optimizer_parameters"] = r["total_params"]
        r["efference_heads"] = {"version": "efference_v1", "weight": 0.1, "tenure_steps": 5,
                                "scales": {"pos_next_cm": 0.452, "rot_next_deg": 2.73}}
        r["contact_head"] = {"version": "contact_head_v1",
                             "patch_names": ["p%d" % i for i in range(53)],
                             "weight": 0.1, "margin": {"self": 0.114, "world": 0.176},
                             "contact_control_weight": 0.5}
        r["motor_synergy"] = {"dim": 16}
        r["vestibular_sense"] = {"dim": 29, "input_bound": 4.0}
        r["wellness_channels"] = {"rank": "normalized", "channels": _GOF_CHANNELS + ["pain"],
                                  "weights": {c: 0.5 for c in _GOF_CHANNELS}, "gate_line": 0.87}
    _edit_receipt(run, rec_edit)
    # the stats: every record gains the blocks
    path = os.path.join(run, "replay_training_stats.jsonl")
    out = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        r = json.loads(line)
        if r.get("type") == "mimo":
            r.update(_gof_record(i, r["step"]))
        out.append(json.dumps(r))
    open(path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    # the snapshots: schema v3 blocks, a self-contact onset, the caregiver's hand-off report
    for name in os.listdir(os.path.join(run, "explain")):
        if not name.startswith("step_"):
            continue
        sp = os.path.join(run, "explain", name)
        snap = json.load(open(sp))
        snap["header"]["schema_version"] = 3
        snap.update(_gof_snapshot_blocks(snap["header"]["step"]))
        snap["senses"]["task"] = {"self_contact_onsets": [
            {"pair": ["left_hand", "left_thdistal"], "regions": ["left hand", "left hand"],
             "structural": False, "in_view": False},
            {"pair": ["left_hand", "left_thhub"], "regions": ["left hand", "left hand"],
             "structural": True, "in_view": False}]}
        snap["caregiver"] = {"mode": "responsive", "object_follows_grip": True,
                             "object_held_by_infant": False, "object_grips": 3}
        snap["motor"].update(order_sent="absent: fixture", rest_window=False)
        json.dump(snap, open(sp, "w"))
    # the causal chain's resolved orders
    with open(os.path.join(run, "causal_chain.jsonl"), "w", encoding="utf-8") as fh:
        for k in range(3):
            fh.write(json.dumps({"type": "latch", "step": 100 + 5 * k, "order_seq": k,
                                 "source": "planner", "bodies": 2, "hold": 5,
                                 "reached_body": True, "plan_share_of_executed": 0.35,
                                 "plan": {"goal": {"value_ranked": [1.0, 1.1]}, "chosen_rank": [1, 2]},
                                 "next": {"pred_err": {"vision": 0.01}, "persist_err": {"vision": 0.02}},
                                 "latch": {"pred_err": {"vision": 0.03}, "persist_err": {"vision": 0.02},
                                           "predicted_wellness_change": 0.001,
                                           "realized_wellness_change": -0.002}}) + "\n")
    # a probe index for this lineage, with one result file whose controls mix both kinds
    pdir = tmp_path / "gof_probes"
    pdir.mkdir()
    res = pdir / "gaze.json"
    json.dump({"controls": {"hold": "every command is compared with holding still",
                            "ablation": {"50ms": {"share_of_error_removed": 5e-8}},
                            "instrument_repeat": {"repeats": 3, "exact": True},
                            "mystery": {"x": 1}}, "status": "measured"}, open(res, "w"))
    head = {"metric": "gaze_captured_share_vision_150ms", "value": 7.3e-8,
            "unit": "share of the real change in the vision token (0-1)", "floor": 0.0,
            "goal": 1.0, "distance_to_goal": 0.9999999,
            "lived": "the picture moved 7.3e-06% of the way"}
    idx = pdir / "index.jsonl"
    with open(idx, "w", encoding="utf-8") as fh:
        for probe, t in (("gaze", "2026-09-22 19:28:29"), ("gaze", "2026-09-22 19:43:41")):
            fh.write(json.dumps({"probe": probe, "item": 7, "run": "H99_fixture",
                                 "checkpoint": "X/ckpt_step_100", "lineage_step": 100, "time": t,
                                 "path": str(res), "headline": head}) + "\n")
        fh.write(json.dumps({"probe": "gaze", "item": 7, "run": "OTHER_lineage",
                             "checkpoint": "Y/ckpt_step_5", "lineage_step": 5,
                             "time": "2026-09-22 19:00:00", "path": str(res),
                             "headline": head}) + "\n")
    monkeypatch.setattr(BX, "PROBE_INDEX", str(idx))
    return run


def test_gof_a_run_with_every_feature_on_draws_each_one(tmp_path, monkeypatch):
    """Every feature the build added reaches the page for a run launched with it: its box,
    its lines, its tour slide, its run-tab charts, its probe and causal-chain readers - and no
    tour slide names a box or a line the diagram does not draw for that run."""
    run = _fixture_run_gof(tmp_path, monkeypatch)
    b = _blocks(BX.render(run))
    meta, series, tour, doc = b["data-meta"], b["data-series"], b["data-tour"], b["data-doc"]
    receipt, args = meta["receipt"], json.load(open(os.path.join(run, "args.json")))
    # the probe index: this lineage only, the newest line per probe, controls in words
    pr = meta["probes"]
    assert [c["checkpoint"] for c in pr["checkpoints"]] == ["ckpt_step_100"]
    g = pr["checkpoints"][0]["probes"]["gaze"]
    assert g["time"] == "2026-09-22 19:43:41"
    assert "every command is compared with holding still" in g["controls"]
    assert any("under 0.001% at 50ms" in c for c in g["controls"]), g["controls"]
    assert any("identical numbers" in c for c in g["controls"])
    assert any("'mystery'" in c for c in g["controls"]), "an unknown control must be named, not dropped"
    # the causal chain's resolved orders, newest last
    assert [e["step"] for e in meta["causal_events"]["events"]] == [100, 105, 110]
    # the liveness measurements come from the repository's own measurement folder
    assert meta["liveness"].get("rows"), meta["liveness"]
    # every build chart is drawn, none is listed as a feature the run lacks
    assert series["off"] == [], series["off"]
    labels = " | ".join(c["label"] for c in series["charts"])
    for needle in ("Causal chain: out of every 100 planning events", "Efference body head", "Efference gaze head",
                   "Vestibular: prediction error", "Controllability", "Learning progress",
                   "Which channels made the goal win", "Contact head against", "Foreseen touches",
                   "Hand pressing", "The machine: memory", "Motor synergy", "Next-frame loss weight",
                   "How much the prediction moves", "World events", "Self-contact per body",
                   "Posture: how far"):
        assert needle in labels, needle
    # the chance line of the causal chain is 1 / (1 + alternatives)
    cc = next(c for c in series["charts"] if c["label"].startswith("Causal chain: out of every 100"))
    chance = next(s for s in cc["series"] if s["key"] == "causal_chain.chance")
    assert all(abs(p[1] - 20.0) < 1e-6 for p in chance["points"]), "the chance line is 1 in 5, drawn per 100"
    # the tour: one slide per feature, naming drawn boxes and animating drawn lines only
    ids = [st["id"] for st in tour]
    for sid in ("entity_self", "vestibular", "efference", "contact_head", "synergy",
                "wellness_channels", "causal_chain"):
        assert sid in ids, sid
    drawable = _drawable_edges(receipt, args)
    nodes = _drawable_nodes(receipt, args)
    for st in tour:
        for a, to in st["edges"]:
            assert (a, to) in drawable, f"slide {st['id']} animates an undrawn line {a}->{to}"
        assert set(st["nodes"]) <= nodes, f"slide {st['id']} names undrawn boxes {set(st['nodes']) - nodes}"
    # the page text is this run's: the sixth slot is the vestibular sense, the toy is handed over
    assert "vestibular" in doc["node.heads"]["w"] and "intermind" not in doc["node.heads"]["w"]
    assert "--toy-hand-off follow" in doc["node.caregiver"]["w"]
    assert "--contact-profile mechanical_v4" in doc["term.contact-profile"]["w"]
    assert "stamps each plan's reached_body at the latch" in doc["node.intention"]["when"]
    assert "--wellness-rank normalized" in doc["term.wellness-ranking"]["w"]
    page = BX.render(run)
    assert not re.findall(r"\{[a-z][a-z0-9_]*\}", re.sub(r"<script>.*?</script>", "", page, flags=re.S))


@pytest.mark.parametrize('planner,babble', [(False, False), (True, False), (False, True)])
def test_gof_the_page_with_every_feature_runs_in_a_headless_browser(tmp_path, monkeypatch, planner, babble):
    """The same page, executed: four tabs, the probes tab drawn, every popup, box, slide and
    snapshot exercised, the new inspector panels drawn, and nothing a reader should never see."""
    CK, exe = _browser_or_skip()
    run = _fixture_run_gof(tmp_path, monkeypatch)
    _edit_receipt(run, lambda r: r['motor_synergy'].update(planner=planner, babble=babble, k=16))
    edge_check = """<script>addEventListener('load',()=>{
      const connected=(a,b)=>!!document.querySelector('.dg-edge[data-a="'+a+'"][data-b="'+b+'"]');
      if(!connected('world','synergy_examples')||!connected('synergy_examples','synergy'))
        throw Error('The synergy learning path is disconnected');
      if(connected('synergy','intention')!==PLANNER)
        throw Error('The synergy control connection contradicts its switch');
      if(connected('synergy','world')!==BABBLE)
        throw Error('The synergy exploration connection contradicts its switch');
    });</script>""".replace('PLANNER', json.dumps(planner)).replace('BABBLE', json.dumps(babble))
    page = tmp_path / "gof.html"
    page.write_text(BX.render(run).replace('</body>', edge_check + '</body>'), encoding="utf-8", newline="")
    rep = CK.check_page(str(page), exe)
    assert rep["ran"], rep.get("why")
    assert not rep["problems"], "\n".join(rep["problems"])
    assert len(rep["tabs"]) == 4 and rep["probes_tab_drawn"] and rep["probe_panels"] == 1
    # the eleven panels of 2026-09-21 plus the causal chain, the efference heads, the wellness
    # channels, foreseen touches and the vestibular vector
    assert rep["inspector_panels"] >= 16, rep["inspector_panels"]
    assert rep["sub_diagrams"] >= 12 and rep["boxes_clicked"] == rep["diagram_boxes"]
    assert rep["popups_opened"] == rep["doc_entries"]
    # DESIGN.md section 11 (operator 2026-09-22): the build's panels and the probes tab print
    # counts, real units or percentages between named ends - never a bare score decimal
    new_panels = ("The causal chain", "What the command was expected", "Balance, neck",
                  "Wellness channels and the goal ranking", "Foreseen touches",
                  "The gain-of-function build's panels not drawn")
    offenders = {w: n for w, n in rep["bare_decimals"]["by_where"].items()
                 if w.startswith("probes tab") or any(t in w for t in new_panels)}
    examples = [e for e in rep["bare_decimals"]["examples"]
                if e["where"].startswith("probes tab") or "inspector" in e["where"]]
    assert not offenders, (offenders, examples[:10])


def test_gof_the_sense_words_follow_the_receipt_and_the_sixth_slot():
    """Sense counts and the sixth slot's name are the receipt's, never typed: with
    --vestibular-sense on the vestibular sense takes intermind's slot, is graded, and is the
    sense the learned wellness copy leaves out."""
    from lem.model.wellness import APPROXIMATE_EXCLUDED
    SCH = BX.SCH
    assert tuple(SCH.APPROXIMATE_EXCLUDED) == tuple(APPROXIMATE_EXCLUDED)
    old = SCH.gof_fill({"active_modalities": SIX}, {"audio": "real"}, "RUNI")
    assert old["{n_senses}"] == "6" and old["{n_senses_word}"] == "six" and old["{n_heads_word}"] == "five"
    assert "intermind is a synthetic placeholder" in old["{graded_sentence}"]
    assert "four of them in RUNI" in old["{graded_text}"]
    assert old["{judge_left_out}"] == "Intermind is the one left out."
    assert "placeholder sense" in old["{hself_slot5_warning}"]
    new = SCH.gof_fill({"active_modalities": _GOF_SENSES}, {"audio": "real"}, "RUNV")
    assert "every one of them measured live" in new["{graded_sentence}"]
    # the registry's own sense names (schema.SENSE_WORDS), never a second name for a sense
    assert new["{head_list}"] == "vision, tactile, audio, proprioception and vestibular"
    assert new["{judge_left_out}"].startswith("Vestibular is left out")
    assert "sixth slot changed meaning" in new["{hself_slot5_warning}"]
    # a snapshot's own source labels win over the launch flags
    lab = SCH.gof_fill({"active_modalities": SIX}, {}, "R", source_mask={"audio": "synthetic"})
    assert "(audio and intermind are synthetic placeholders" in lab["{graded_sentence}"] and "three of the five" in lab["{graded_sentence}"]
    # a flag the launch does not record says so, never another run's setting
    none = SCH.gof_fill({}, {}, "RUNN")
    assert "do not record --toy-hand-off" in none["{toy_hand_off_run}"]
    assert "before --causal-chain existed" in none["{reached_body_sentence}"]
    assert "not yet in a running segment" in none["{synergy_run}"]
    # the DOC carries the placeholders, not the numbers
    assert "{n_heads_word} in {run_name}: {head_list}" in SCH.DOC["node.heads"]["w"]
    assert "five here: vision, touch, audio, proprioception, intermind" not in SCH.DOC["node.heads"]["w"]


def test_gof_every_published_flag_is_one_the_trainer_parses():
    """PUBLISHED_ARGS gained the build's flags and the world settings the survey found
    unpublished (stale item 13); each must be a flag train_mimo really takes, or the page
    prints 'not recorded' for a setting that was never a setting."""
    from mimo_bridge import train_mimo as TM
    known = set(vars(TM.parse_args(["--out", "unused"])))
    missing = sorted(set(BX.PUBLISHED_ARGS) - known)
    assert not missing, f"PUBLISHED_ARGS names flags train_mimo does not parse: {missing}"
    for k in ("contact_profile", "toy_hand_off", "finger_armature", "hinge_armature",
              "joint_damping_scale", "entity_self", "entity_self_teacher", "self_value_weight",
              "self_value_level", "self_value_gate_windows", "gaze_kp", "gaze_kd", "vision_gate",
              "body_function", "causal_chain", "efference_heads", "vestibular_sense",
              "wellness_rank", "controllability", "learning_progress", "curiosity_mode",
              "consequence_self_contact", "contact_head", "contact_control_wellness",
              "motor_synergy", "smooth_source"):
        assert k in BX.PUBLISHED_ARGS, k


def test_gof_the_surveys_stale_items_are_fixed():
    """docs/gain_of_function_v3/agents/understand-explainer.md section 3, items 1-18: each
    stale sentence is gone from the page's sources and the true one is there."""
    import inspect
    from mimo_bridge.explainer import capture as CAP
    from mimo_bridge.explainer import walkthrough as WT
    from mimo_bridge.touch_adapter import PUSH_ARRIVAL_SCALE_CMS, PUSH_DEPTH_SCALE_CM
    from mimo_bridge.lem_model import TOUCH_TOKEN_F
    doc, js = _page_doc(), _viewer_js()
    flat = _all_doc_text(doc)
    # 1 and 2: the tactile row's width
    assert "32 rows × 12 scaled fields" not in js and "touchLegend(s)" in js
    assert (f"the bridge's {TOUCH_TOKEN_F}-wide tactile row keeps the first 8"
            in doc["node.sense.proprioception"]["f"])
    # 3: the push's scales are the adapter's own
    enc = doc["edge.sense_tactile.enc"]["w"]
    # DESIGN.md section 11 (2026-09-23): the depth divisor is written in words; the words are
    # held to the code's own constant here
    assert PUSH_DEPTH_SCALE_CM == 0.1 and "centimetres divided by a tenth of a centimetre" in enc
    assert f"centimetres a second divided by {PUSH_ARRIVAL_SCALE_CMS:g}" in enc
    assert "centimetres over 10" not in enc and "over 500)" not in enc
    # 4: the toy hand-off
    assert "toy-hand-off" in doc["node.caregiver"]["w"] and "a push cannot move it" in doc["node.caregiver"]["w"]
    assert "the infant's hand cannot displace the toy at all" not in doc["incorporation"]["w"]
    assert "under the toy hand-off it goes" in js
    # 5: the contact profile is the run's, never 'what the crib runs'
    assert "WHAT THE CRIB RUNS" not in flat and "No run uses v3 or v4 yet" not in flat
    assert "{contact_profile_run}" in doc["term.contact-profile"]["w"]
    assert "the signal wellness reads today on the live path" not in js
    # 6: fatigue carries its measured anchors
    assert "written-down guess" not in doc["term.muscle-groups"]["h"]
    assert "no_gear" in doc["term.muscle-groups"]["f"] and "not mechanical power in watts" in doc["term.muscle-groups"]["f"]
    # 7 and 8: the tour's wellness and curriculum slides
    steps = {st["id"]: st["body"] for st in WT.build_steps(
        {"active_modalities": SIX, "forward_model": {"_seq_length": 24, "_n_modalities": 6}}, {})}
    assert steps["wellness"].startswith("Fixed formulas calculate six scores per body")
    assert "Five grounded evaluators" not in steps["wellness"]
    assert "40 ms to 0" not in steps["curriculum"] and "no sense is delayed" in steps["curriculum"]
    # 9: the wellness chart draws self-touch
    recs = [{"step": i, "wellness": {"self_touch": 0.0, "coherence": 0.3}} for i in range(4)]
    wc = next(c for c in BX.build_series({"mimo": recs, "training": [], "launch": []}, [])["charts"]
              if c["doc"] == "chart.wellness")
    assert "wellness.self_touch" in [s["key"] for s in wc["series"]]
    assert "each evaluator scoring 0 to 1" not in doc["chart.wellness"]["w"]
    # 10: radians are never shown alone
    bx_src = open(os.path.join(os.path.dirname(BX.PKG), "tools", "build_explainer.py"),
                  encoding="utf-8").read()
    assert "(rad)" not in js and "command size (radians)" not in bx_src
    recs = [{"step": i, "cmd_mag_rad": 0.5} for i in range(4)]
    c5 = next(c for c in BX.build_series({"mimo": recs, "training": [], "launch": []}, [])["charts"]
              if c["doc"] == "chart.cmd_follow")
    deg = next(s for s in c5["series"] if s["key"] == "cmd_mag_deg")
    assert abs(deg["points"][0][1] - 0.5 * 180 / 3.141592653589793) < 1e-3
    # 11: the ceiling carries the reader and an added curiosity (capture's parts)
    cap = open(os.path.join(BX.PKG, "capture.py"), encoding="utf-8").read()
    assert '"reader": reader_ceiling(model)' in cap and '"curiosity_added"' in cap
    # 12: the named-things box is drawn when the run has it
    assert '["entity_self", "named things (dark) + reader"' in js
    # 14: the finger armature is described
    assert "term.finger-armature" in doc and "[[finger-armature" in flat
    # 15: the capture takes the order really sent and the rest window
    params = inspect.signature(CAP.build_snapshot).parameters
    assert "order_sent" in params and "rest_window" in params
    # 16: node.core says when it runs; no 'lite' wording in the tour
    assert "when" in doc["node.core"]
    for st in WT.build_steps({"active_modalities": SIX}, {}):
        assert "current lite" not in st["body"] and "deliberately small 'lite'" not in st["body"]
    # latent: reached_body is filled under --causal-chain, and the page says so
    assert "nothing in the code ever fills" not in flat and "never filled" not in js
    # every drawn box's entry says when it runs
    assert not [k for k in doc if k.startswith("node.") and "when" not in doc[k]]


def test_gof_a_chart_whose_feature_is_off_is_named_apart_from_thin_data():
    """A log without a build's block lists the chart under 'feature off'; a log with the block
    but too few points lists it under 'not drawn'; a log with it draws it."""
    base = [{"step": i, "causal_pe": 0.02} for i in range(6)]
    out = BX.build_series({"mimo": base, "training": [], "launch": []}, [])
    assert any("Causal chain" in x for x in out["off"])
    assert not any("Causal chain" in x for x in out["dropped"])
    thin = [dict(r, causal_chain={"events": 0}) for r in base]
    out = BX.build_series({"mimo": thin, "training": [], "launch": []}, [])
    assert any("Causal chain: out of every 100" in x for x in out["dropped"])
    full = [dict(r, **_gof_record(i, r["step"])) for i, r in enumerate(base)]
    out = BX.build_series({"mimo": full, "training": [], "launch": []}, [])
    assert out["off"] == [] and not any("Causal chain" in x for x in out["dropped"])
    assert "SERIES.off" in _viewer_js()


def test_gof_the_liveness_reader_reads_the_measurement_files(tmp_path, monkeypatch):
    """The liveness table is read from the lead's measurement files; a missing folder says so."""
    live = BX.read_liveness()
    assert [c["key"] for c in live["columns"]][:2] == ["pose-head", "wellness-copy"]
    assert live["rows"][-1]["step"] == 344400 and live["rows"][-1]["values"][0] == "0.000"
    assert live["sweep"]["dead_layers"]["tenure_heads.pose.0"] == 0.0
    assert "encoder_registry.encoders.motor.pool_attn" in live["sweep"]["locked_attention"]
    monkeypatch.setattr(BX, "LIVENESS_DIR", str(tmp_path / "nowhere"))
    assert "absent" in BX.read_liveness()["missing"]
    assert BX.lineage_prefix("H84_fresh_drivers_20260918_seg31_toy_and_push") == "H84_fresh_drivers_20260918"


def test_gof_the_builds_charts_and_controls_print_counts_units_or_percentages():
    """DESIGN.md section 11 (operator 2026-09-22): a number a person reads is a count, a real
    unit, or a percentage between named ends. Every chart of the build names the unit its ticks
    carry, no cosine or correlation is printed as a number - since 2026-09-23 one may be DRAWN, on
    a chart that carries its named ends and no numeric tick at all - and the probe controls are
    written in per cent - a tiny share as 'under 0.001%', never an exponent."""
    base = [dict({"step": i}, **_gof_record(i, i)) for i in range(6)]
    out = BX.build_series({"mimo": base, "training": [], "launch": []}, [])
    counts_only = {"chart.self_contact", "chart.entity_self", "chart.liveness"}
    build_docs = {k for k in BX.DOC_CHARTS if k in (
        "chart.causal_choice", "chart.causal_prediction", "chart.causal_wellness",
        "chart.efference_body", "chart.efference_gaze", "chart.vestibular",
        "chart.vestibular_posture", "chart.posture", "chart.controllability",
        "chart.controllability_gates",
        "chart.learning_progress", "chart.wellness_rank", "chart.contact_head",
        "chart.contact_control", "chart.self_contact_share", "chart.hand_press",
        "chart.consequence", "chart.incorporation", "chart.wellness_align",
        "chart.prediction_audit", "chart.order_sensitivity", "chart.skin_regions",
        "chart.world_events", "chart.machine_memory", "chart.machine_time", "chart.synergy",
        "chart.next_frame_norm")}
    drawn = [c for c in out["charts"] if c["doc"] in build_docs | counts_only]
    assert drawn
    for c in drawn:
        keys = " ".join(s["key"] for s in c["series"])
        if c.get("ends"):
            # a picture: named ends, no unit, so no tick and no hover prints a number
            assert not c.get("unit"), f"chart {c['label']!r} both draws named ends and prints a unit"
            assert all(isinstance(e[1], str) and e[1] for e in c["ends"])
            continue
        assert c.get("unit"), f"chart {c['label']!r} names no unit for its ticks"
        for banned in ("agreement_mean", "control_mean", "wellness_change_corr", "following_mean",
                       "difference_mean", "liveness.pools"):
            assert banned not in keys, f"chart {c['label']!r} prints {banned}, a cosine, correlation or entropy"
    # the item-5 measurement is drawn, never printed; the explained share is logged beside it
    syn = [c for c in drawn if c["doc"] == "chart.synergy"]
    assert any(c.get("ends") and "synergy.following_mean" in [s["key"] for s in c["series"]] for c in syn)
    # a share drawn per 100: the chance line of four alternatives is 20 in 100
    cc = next(c for c in drawn if c["doc"] == "chart.causal_choice")
    best = next(s for s in cc["series"] if s["key"] == "causal_chain.chosen_best_per100")
    assert abs(best["points"][0][1] - 20.0) < 1e-6
    assert BX._pct_text(5e-8) == "under 0.001%" and BX._pct_text(0.123) == "12%"
    assert BX._pct_text(0.0456) == "4.6%" and BX._pct_text(-0.2) == "-20%"
    assert BX._pct_text(None) == "not recorded"


def test_gof_foreseen_touches_set_the_control_beside_the_same_onsets():
    """fix-readout-explainer 2026-09-23. The contact head's success_self counts every scored
    self-touch onset; success_control counts only the onsets of the moments the control ran (every
    control_period-th moment, two or more bodies). The chart drew those two as its pair, so their
    gap mixed the order's effect with WHICH onsets were counted. The producer's same-onset pair is
    success_paired (this body's own order on the control's onsets) against success_control.

    The producer, on a window where the own order foresees 12 of 20 onsets overall but 1 of 4 on
    the control's onsets, and another body's order also 1 of 4: the old pair reads 60 against 25,
    as if the order added 35 foreseen touches in 100; the same onsets read 25 against 25."""
    from lem.model.contact_head import ContactTracker
    tr = ContactTracker()
    tr.win.update(scorable_self=20, predicted_self=12, ctl_steps=2, ctl_scorable_self=4,
                  ctl_real_pred_self=1, ctl_pred_self=1, scorable_world=10, predicted_world=5,
                  ctl_onsets_world=5, ctl_real_pred_world=3, ctl_pred_world=1)
    s = tr.summary()
    assert (s["success_self"], s["success_paired"], s["success_control"]) == (0.6, 0.25, 0.25)
    assert (s["success_world"], s["success_paired_world"], s["success_control_world"]) == (0.5, 0.6, 0.2)
    fields = ("success_self", "success_paired", "success_control", "success_world",
              "success_paired_world", "success_control_world")
    base = [dict({"step": i}, **_gof_record(i, i)) for i in range(6)]
    for r in base:
        r["contact_control"] = dict(r["contact_control"], **{k: s[k] for k in fields})
    out = BX.build_series({"mimo": base, "training": [], "launch": []}, [])
    c = next(c for c in out["charts"] if c["doc"] == "chart.contact_control")
    keys = [x["key"] for x in c["series"]]
    lab = {x["key"]: x["label"] for x in c["series"]}
    val = {x["key"]: x["points"][0][1] for x in c["series"]}
    P = "contact_control."
    # the control is drawn right after the own-order line counted on the same onsets
    i = keys.index(P + "success_control_per100")
    assert keys[i - 1] == P + "success_paired_per100", keys
    assert "control's onsets" in lab[P + "success_paired_per100"] and "same onsets" in lab[P + "success_control_per100"]
    assert abs(val[P + "success_paired_per100"] - 25.0) < 1e-9 and abs(val[P + "success_control_per100"] - 25.0) < 1e-9
    j = keys.index(P + "success_control_world_per100")
    assert keys[j - 1] == P + "success_paired_world_per100", keys
    # the every-onset lines say so, and neither is labelled as the control's pair
    assert "every scored onset" in lab[P + "success_self_per100"] and "every scored onset" in lab[P + "success_world_per100"]
    assert abs(val[P + "success_self_per100"] - 60.0) < 1e-9
    # the popup names the same-onset pair and the mistake it replaces
    doc = BX.DOC_CHARTS["chart.contact_control"]
    assert "success_paired against success_control" in doc["h"]
    assert "setting success_self beside success_control" in doc["f"]
    # a log written before success_paired existed draws no pair, and its own-order line is still
    # the every-onset one
    old = [dict(r, contact_control={k: v for k, v in r["contact_control"].items()
                                    if not k.startswith("success_paired")}) for r in base]
    c = next(c for c in BX.build_series({"mimo": old, "training": [], "launch": []}, [])["charts"]
             if c["doc"] == "chart.contact_control")
    assert P + "success_paired_per100" not in [x["key"] for x in c["series"]]


def test_the_liveness_texts_say_what_the_line_means_without_a_raw_input_value():
    """fix-readout-explainer 2026-09-23. The term popup, the liveness chart's popup and the probes
    tab's caption said "input above -3", a raw input value no reader can place (the extended
    scanner's rule, DESIGN.md section 11). They now say what the line means: below it GELU moves a
    unit's output by less than 1.2% of a change in its input - GELU's slope at the probe harness's
    line, computed, so a changed line cannot leave the page's share behind."""
    from mimo_bridge.tools import gof_probe_harness as H
    from mimo_bridge.tools.check_no_bare_decimals import readout_problems
    share = H.pct(H.LIVE_FOLLOW, 2)
    assert share == "1.2%"
    texts = {"term.liveness": " ".join(BX.SCH.DOC["term.liveness"].values()),
             "chart.liveness": " ".join(BX.DOC_CHARTS["chart.liveness"].values())}
    js = _viewer_js()
    k = js.index("A unit counts as live when")
    texts["probes tab caption"] = re.sub(r'"\s*\+\s*"', "", js[k:js.index("}));", k)])
    for name, t in texts.items():
        assert share + " of the change" in t, (name, t)
        assert not readout_problems(t), (name, readout_problems(t))
        assert "above -3" not in t and "below -3" not in t, name


def test_no_popup_prints_a_raw_range_of_internal_values():
    """verify-fix-readout-explainer 2026-09-23. A signed range with no unit after it ("unitless
    internal features, typically -3 to 3", the edge.ln_out.td_gate popup) is a raw internal value no
    reader can place (DESIGN.md section 11), of the same kind as "inputs from -32.9 to -7.1"; the
    extended scanner reads a range only after the word "input", so it passed. It was never measured
    either. No popup of the page, term or chart, prints one."""
    from mimo_bridge.tools.check_no_bare_decimals import _unit_follows
    signed_range = re.compile(r"(?<![\w.\-−])[-−]\d+(?:\.\d+)?\s*(?:to|–|—)\s*[-−]?\d+(?:\.\d+)?")
    found = []
    for src, docs in (("DOC", BX.SCH.DOC), ("DOC_CHARTS", BX.DOC_CHARTS)):
        for key, entry in docs.items():
            for field, text in entry.items():
                for m in signed_range.finditer(str(text)):
                    if not _unit_follows(str(text), m.end()):
                        found.append((src, key, field, m.group(0)))
    assert not found, found


# ================================================================= DESIGN.md section 11 on the whole page
# (fix-explainer, 2026-09-23). The operator: "make sure the dashboard and the explainer don't use
# decimals like this". The page check hands back every text the exercise renders - tabs, every popup,
# every box, every sub-diagram, EVERY tour slide, every snapshot in the inspector and what a hover
# shows on the run tab and on the heatmaps - and scans it with tools/check_no_bare_decimals.py, the
# same rule the dashboards and the watcher lines are held to.
def _page_bare_decimals(rep):
    bd = rep["bare_decimals"]
    return bd["count"], [(h["where"], h["number"], h["context"]) for h in bd["hits"][:15]]


def _replace_block(page, bid, obj):
    head = f'<script type="application/json" id="{bid}">'
    start = page.index(head) + len(head)
    end = page.index("</script>", start)
    return page[:start] + json.dumps(obj, separators=(",", ":")).replace("</", "<\\/") + page[end:]


def test_s11_the_newest_real_runs_page_shows_no_bare_score_decimal(tmp_path):
    """The page built for the newest run under Saved - on 2026-09-23 H84 segment 31, whose newest
    snapshot is step_343139.json (2026-09-22 16:19) - read by the headless check: 0 bare score
    decimals anywhere a person reads (1,865 by this scanner before the pass of 2026-09-23; 2,079 by
    the looser in-page count it replaced). A launch setting written the way the command line writes
    it ("--babble-noise 0.05") is the only form excused, and it is counted apart."""
    CK, exe = _browser_or_skip()
    run = BX.newest_run()
    if not run or not os.path.isdir(os.path.join(run, "explain")):
        pytest.skip("no run with snapshots under Saved/mimo_bridge on this checkout")
    page = tmp_path / "real_s11.html"
    page.write_text(BX.render(run), encoding="utf-8", newline="")
    newest = BX.snapshot_entries(run)[-1]["step"]
    meta = _blocks(page.read_text(encoding="utf-8"))["data-meta"]
    assert newest in meta["snapshots"]["embedded_steps"], "the page must embed the newest snapshot"
    rep = CK.check_page(str(page), exe)
    assert rep["ran"], rep.get("why")
    assert not rep["problems"], "\n".join(rep["problems"])
    n, hits = _page_bare_decimals(rep)
    assert n == 0, (f"{n} bare score decimals on the page of {os.path.basename(run)} "
                    f"(newest snapshot step {newest}): {hits}")
    assert rep["hovered"] > 0, "no hover was read: the readout and the heatmaps went unchecked"


def test_s11_the_fixture_pages_with_every_feature_show_no_bare_score_decimal(tmp_path, monkeypatch):
    """The same on the two fixture pages, which carry what a real run may not: the hierarchical
    self's panel, the causal chain, the efference heads, the wellness channels, foreseen touches, the
    vestibular vector, the goal the planner chose, every build chart and a probe sentence written in
    exponent form."""
    CK, exe = _browser_or_skip()
    (tmp_path / "older").mkdir()
    (tmp_path / "gof").mkdir()
    run = _fixture_run(tmp_path / "older")
    _set_args(run, top_down_gate="on", hierarchical_self="dark", hself_slots=64,
              caregiver_mode="on", persistent_memory="on", self_state="learn",
              intention="on", intention_every=10, imagination="on", imagination_steps=3)
    gof = _fixture_run_gof(tmp_path / "gof", monkeypatch)
    for name, r in (("older", run), ("gof", gof)):
        page = tmp_path / f"{name}.html"
        page.write_text(BX.render(r), encoding="utf-8", newline="")
        rep = CK.check_page(str(page), exe)
        assert rep["ran"] and not rep["problems"], rep.get("problems")
        n, hits = _page_bare_decimals(rep)
        assert n == 0, (name, n, hits)


def test_s11_the_page_check_reads_every_slide_and_popup_and_can_fail(tmp_path):
    """The control. A decimal planted in a tour slide that is not the last one and one planted in a
    popup are each found; the check that read the LAST tour slide only (before 2026-09-23) could not
    see the first. A launch setting in command-line form is excused and counted apart."""
    CK, exe = _browser_or_skip()
    run = _fixture_run(tmp_path)
    good = BX.render(run)
    b = _blocks(good)
    tour, doc = b["data-tour"], b["data-doc"]
    assert len(tour) > 2
    tour[0]["body"] += " A planted score of 0.37 on the first slide."
    doc["term.token"]["w"] += " A planted score of 0.42 in a popup. Launched with --babble-noise 0.05."
    bad = _replace_block(_replace_block(good, "data-tour", tour), "data-doc", doc)
    page = tmp_path / "planted.html"
    page.write_text(bad, encoding="utf-8", newline="")
    rep = CK.check_page(str(page), exe)
    assert rep["ran"], rep.get("why")
    found = {(h["number"], h["where"].split(" ")[0]) for h in rep["bare_decimals"]["hits"]}
    assert ("0.37", "tour") in found, found
    assert ("0.42", "popup") in found, found
    assert any(s["setting"] == "--babble-noise 0.05" for s in rep["bare_decimals"]["settings_examples"])
    # the scanner is the section 11 rule itself, not a copy of it
    assert CK.scan_rendered_texts([["x", "a score of 0.14 and 12 cm and 0.3 cm"]])["count"] == 1


def test_the_page_check_holds_the_scanners_rules_for_numbers_of_any_size():
    """fix-readout-explainer 2026-09-23: the page check scans with readout_problems, so a score of 1
    or more with no unit, a multiplier with nothing named after it and a raw hidden-unit input value
    fail the page as a bare decimal does (each sentence is one the scanner's author found printed).
    A launch setting in command-line form stays excused, and a count or a unit passes."""
    from mimo_bridge.tools import check_explainer_page as CK
    rep = CK.scan_rendered_texts([
        ["popup a", "the posture probe's error ratio 1.003 on the head commands"],
        ["popup b", "ungraded-channel drift x1.0. The window closed."],
        ["probes tab", "a unit responds when its input rises above -3; the highest input any unit received was -7"],
        ["popup c", "a score of 0.14"],
        ["popup d", "launched with --babble-noise 0.05; 4.2 times its first value; 32 of every 100 "
                    "commands; 12 cm; the output moves by 2 in every hundred billion parts of a change "
                    "in its input"]])
    assert rep["by_rule"] == {"score": 1, "multiplier": 1, "unit input": 2, "bare decimal": 1}, rep["by_rule"]
    assert rep["count"] == 5 and "popup d" not in rep["by_where"], rep["hits"]
    assert all(h.get("rule") for h in rep["hits"]) and rep["settings_excused"] == 1


def test_s11_every_chart_says_how_its_numbers_are_read():
    """build_series refuses a chart that names neither a unit nor its named ends, so no chart can
    reach the page printing raw numbers on its ticks; the older charts carry a view (scale,
    relative) and the logged series keep the run's own values, which the verdict tiles read."""
    src = open(BX.__file__, encoding="utf-8").read()
    assert "needs unit= or ends= (DESIGN.md section 11)" in src
    recs = [{"type": "mimo", "step": 100 * (i + 1), "causal_pe": 0.02,
             "tenure_tok": {"rank1": 0.2, "n_decoys": 14.0},
             "tenure_gap": {"pose": 0.01}, "tenure_pe": {"pose": 0.03},
             "cmd_response_cos": 0.1, "cmd_response_cos_held": 0.15} for i in range(6)]
    out = BX.build_series({"mimo": recs, "training": [], "launch": []}, [])
    for c in out["charts"]:
        assert c.get("unit") or c.get("ends"), c["label"]
    head = _chart(out, "Does the command explain")
    assert head["scale"] == 100.0 and head["unit"] == "/100"
    assert head["series"][0]["points"][0][1] == 0.2, "the logged value stays the log's own"
    gap = _chart(out, "Is knowing the command worth")
    assert [s["key"] for s in gap["series"]] == ["tenure_gap.share_removed"]
    assert abs(gap["series"][0]["points"][0][1] - 0.25) < 1e-9, "0.01 removed of a do-nothing error of 0.04"
    follow = _chart(out, "Do the limbs obey")
    assert [e[1] for e in follow["ends"]] == ["exactly as commanded", "unrelated to the command",
                                              "exactly against it"]
    assert not follow.get("unit")
    js = _viewer_js()
    assert "function chartView(chart)" in js and "function sayChartValue(chart, v)" in js
    assert "fmt(best[1], 5)" not in js, "the run tab's hover prints the raw logged number again"


def test_s11_the_viewer_never_writes_an_exponent_or_a_raw_internal_as_a_number():
    """fmt() used to write 1.0e-5 for a small number; it now says 'under' the step its decimals can
    show. The readers the older panels use exist, and a picture prints no digit of its value."""
    js = _viewer_js()
    body = js[js.index("function fmt(x, d)"):js.index("function isObj(")]
    assert "toExponential" not in body
    for reader in ("function tokErr(", "function tokSize(", "function typMiss(", "function blend(",
                   "function picture(", "function linkPicture(", "function rawDisp(",
                   "function goalRows(", "function plainOr(", "function plainExponents("):
        assert reader in js, reader
    pic = js[js.index("function picture("):js.index("function linkPicture(")]
    assert "m.style.left" in pic and "NOT_COUNT" in pic and "String(v)" not in pic


# ================================================================= review b1, minor 5 (2026-09-22)
def test_b1_minor5_the_gaze_head_and_the_clipping_sets_follow_the_runs_own_flags():
    """The page said the gaze head's weights 'never move at all' and that gradient clipping is
    'split in two'. With --efference-heads on both are false: the efference gaze head reuses the
    GazeHead's layers and trains them, and the efference heads (with those layers), the contact
    head and the synergy are each clipped as a set of their own (LemLite._clip_training_gradients)."""
    import inspect
    from mimo_bridge.lem_model import LemLite
    clip_src = inspect.getsource(LemLite._clip_training_gradients)
    for part in ("self.efference.parameters()", "self.liveness.parameters()",
                 "self.contact_head.parameters()", "self.synergy.parameters()",
                 "self.self_state_mode == 'learn'"):
        assert part in clip_src, f"the clipping sets changed ({part}): re-word schema.clip_groups"
    comps = {"self_state": {"parameters": 1000, "parameter_tensors": 10},
             "gaze_head": {"parameters": 300, "parameter_tensors": 3},
             "efference_heads": {"parameters": 500, "parameter_tensors": 5},
             "contact_head": {"parameters": 90, "parameter_tensors": 4}}
    base = {"self_state": "learn", "total_params": 10000,
            "component_params": {"voice_head": 200, "gaze_head": 300, "self_state": 1000},
            "voice_head": {"enabled": True, "params": 200},
            "checkpoint_inventory": {"optimizer_parameters": 10000, "optimizer_parameter_tensors": 100,
                                     "components": comps}}
    old = BX.SCH.gof_fill(base, {}, "RUNOLD")
    assert old["{clip_sets_text}"].startswith("two times, over two separate sets")
    assert "the rest of the model's 9,000 numbers in 90 tensors" in old["{clip_sets_text}"]
    assert "the voice head's 200 and the gaze head's 300" in old["{idle_weights_sentence}"]
    assert "nothing reads its output" in old["{gaze_head_use}"]
    new = dict(base, efference_heads={"version": "efference_v1"}, contact_head={"version": "c"})
    eff = BX.SCH.gof_fill(new, {}, "RUNEFF")
    assert eff["{clip_sets_text}"].startswith("four times, over four separate sets")
    assert "the efference heads' (with the GazeHead layers they repurpose) 800 numbers in 8 tensors" \
        in eff["{clip_sets_text}"]
    assert "the rest of the model's 8,110 numbers" in eff["{clip_sets_text}"]
    assert eff["{idle_weights_sentence}"].startswith("200 weights never move at all: the voice head's 200 sit")
    assert "The gaze head's 300 numbers DO move in RUNEFF" in eff["{idle_weights_sentence}"]
    assert "reuses its attention and layers as the efference gaze head" in eff["{gaze_head_use}"]
    current = dict(new, efference_heads={"version": "efference_v2", "gaze_head_repurposed": False},
                   liveness_repair={"mode": "on"})
    current['checkpoint_inventory'] = dict(base['checkpoint_inventory'],
        components=dict(comps, liveness_repair={"parameters": 200, "parameter_tensors": 2}))
    groups = BX.SCH.clip_groups(current)
    assert groups['n_sets'] == 5
    assert groups['main_numbers'] == 8210
    assert groups['groups'][1]['counts'] == (500, 5)
    # and perceive mode is ONE set: self perception joins the rest
    per = BX.SCH.gof_fill(dict(base, self_state="perceive"), {}, "RUNPER")
    assert per["{clip_sets_text}"] == "once, over every optimized number together"
    doc = BX.SCH.DOC
    assert "main gradient groups are clipped" in doc["bp.backward"]["w"]
    assert "after the main clipping step" in doc["bp.backward"]["w"]
    assert "saved optimizer state" in doc["bp.update"]["w"]
    assert "never move at all" not in doc["bp.update"]["w"]
    assert "{gaze_head_use}" in doc["node.gaze"]["w"] and "vestigial" not in doc["node.gaze"]["h"]
    js = _viewer_js()
    # the footer reads the receipt's own gaze_head_repurposed (verify-fix-explainer, 2026-09-23)
    assert "effStamp.gaze_head_repurposed !== false" in js and "reused as the efference gaze head" in js


# ================================================================= review b1, minor 7 (2026-09-22)
def test_b1_minor7_the_causal_events_reader_follows_a_rotated_file(tmp_path):
    """causal_chain.py renames causal_chain.jsonl to causal_chain_<time>.jsonl past its size limit;
    right after a rotation the current file holds a handful of lines, so the reader reads the newest
    rotated file too and keeps the newest resolved orders of the two."""
    run = tmp_path / "R"
    run.mkdir()

    def line(step):
        return json.dumps({"type": "latch", "step": step, "plan": {"chosen_rank": [1]}}) + "\n"
    (run / "causal_chain_20260922_120000.jsonl").write_text("".join(line(s) for s in range(10, 60, 5)))
    (run / "causal_chain.jsonl").write_text(line(60))
    out = BX.read_causal_events(str(run))
    steps = [e["step"] for e in out["events"]]
    assert steps[-1] == 60 and 55 in steps and len(steps) == BX.CAUSAL_EVENTS_MAX
    assert out["files"] == ["causal_chain_20260922_120000.jsonl", "causal_chain.jsonl"]
    only_rotated = tmp_path / "S"
    only_rotated.mkdir()
    (only_rotated / "causal_chain_20260922_120000.jsonl").write_text(line(5))
    assert [e["step"] for e in BX.read_causal_events(str(only_rotated))["events"]] == [5]
    assert "missing" in BX.read_causal_events(str(tmp_path))


# ================================================================= verify-fix-explainer (2026-09-23)
# The adversarial check of fix-explainer's commit ddf4d3922. Each test failed on that commit.
def test_vfe_the_gaze_sentences_read_the_receipts_own_repurpose_stamp():
    """b1 minor 5's page half went by 'are the efference heads on?' alone. LemLite stamps whether the
    heads reuse the GazeHead's layers (receipt efference_heads.gaze_head_repurposed; False for heads
    built with a gaze pool of their own - the review's third remedy). With the stamp False the page
    must not say the GazeHead's numbers move."""
    import inspect
    # efference_v2 (lane-b1-efference, 2026-09-23): the gaze head has layers of its own, so the stamp
    # lem.model.efference.stamp writes is False; a load keeps True for a GazeHead v1 retrained
    from lem.model import efference as EF
    assert '"gaze_head_repurposed": False' in inspect.getsource(EF.stamp)
    comps = {"gaze_head": {"parameters": 300, "parameter_tensors": 3},
             "efference_heads": {"parameters": 500, "parameter_tensors": 5}}
    base = {"total_params": 10000, "component_params": {"voice_head": 200, "gaze_head": 300},
            "voice_head": {"enabled": True, "params": 200},
            "checkpoint_inventory": {"optimizer_parameters": 10000, "optimizer_parameter_tensors": 100,
                                     "components": comps}}
    own = BX.SCH.gof_fill(dict(base, efference_heads={"gaze_head_repurposed": False}), {}, "RUNOWN")
    assert "DO move" not in own["{idle_weights_sentence}"]
    assert "the gaze head's 300" in own["{idle_weights_sentence}"], "an unused GazeHead is idle"
    assert "layers of its own" in own["{gaze_head_use}"]
    assert "the efference heads' 500 numbers" in own["{clip_sets_text}"]
    assert "the rest of the model's 9,500 numbers" in own["{clip_sets_text}"]
    reused = BX.SCH.gof_fill(dict(base, efference_heads={"gaze_head_repurposed": True}), {}, "RUNRE")
    assert "The gaze head's 300 numbers DO move in RUNRE" in reused["{idle_weights_sentence}"]
    # a stamp written before the field existed: every such build reused the layers
    assert BX.SCH.gaze_head_reused({"efference_heads": {"version": "efference_v1"}})
    assert not BX.SCH.gaze_head_reused({})
    js = _viewer_js()
    assert "effStamp.gaze_head_repurposed !== false" in js


def test_vfe_no_page_text_says_a_gated_moment_is_never_chosen_without_the_fallback():
    """The page said a moment over the integrity gate line 'is never chosen' / 'never a goal'. The
    code gives a body whose EVERY remembered moment is over the line its least harmful one
    (rank_values; the review of b3 found the same false sentence in the trainer's help text). Every
    such sentence now carries the condition, and the gate line is the module's own."""
    import inspect
    import torch
    from types import SimpleNamespace
    from mimo_bridge.explainer import walkthrough as WT
    from lem.model import wellness_channels as WCM
    # the code: two moments, both over the line - the less harmful one is still a goal
    line = float(WCM.INTEGRITY_GATE_LINE)
    gate_row = getattr(WCM, "HARM_ROW", WCM.PAIN_ROW)
    names = ["coherence", gate_row] + ([WCM.PAIN_ROW] if gate_row != WCM.PAIN_ROW else [])
    ch = torch.tensor([[[1.0], [line + 0.05]] + ([[line + 0.05]] if len(names) == 3 else []),
                       [[0.0], [line + 0.02]] + ([[line + 0.02]] if len(names) == 3 else [])])
    ranked, _, gated, all_gated = WCM.rank_values(
        ch, names, {"coherence": 1.0}, SimpleNamespace(centre_scale=lambda n: (0.0, 1.0)), {}, line)
    assert bool(all_gated.all()) and bool(gated.all())
    assert int(ranked[:, 0].argmax()) == 1 and torch.isfinite(ranked[1, 0]), \
        "the fallback is gone: re-word the page's gate sentences back"
    # the page: no 'never chosen' / 'never a goal' without 'while any ... is under it'
    sources = {"schema DOC": json.dumps(BX.SCH.DOC), "chart DOC": json.dumps(BX.DOC_CHARTS),
               "viewer.js": _viewer_js(), "walkthrough.py": inspect.getsource(WT)}
    bad = re.compile(r"never (?:be )?(?:chosen|a goal)(?![^.;)]{0,40}while any)")
    for where, text in sources.items():
        hits = [text[max(0, m.start() - 80):m.end() + 40] for m in bad.finditer(" ".join(text.split()))]
        assert not hits, (where, hits)
    fill = BX.SCH.integrity_gate_fill()["{integrity_gate_text}"]
    assert f"{round(100 * line, 1):g}% of the way" in fill and "least harmful" in fill
    assert "{harm_line_pct}" in BX.SCH.DOC["term.integrity"]["w"]
    assert "least harmful remains available" in BX.SCH.DOC["term.integrity"]["w"]
    assert "87%" not in BX.SCH.DOC["term.integrity"]["w"], "the line is the module's, never typed"
    out = BX.build_series({"mimo": [dict({"step": i}, **_gof_record(i, i)) for i in range(6)],
                           "training": [], "launch": []}, [])
    wr = _chart(out, "Which channels made the goal win")
    assert "wellness_rank.all_gated_per100" in [s["key"] for s in wr["series"]]


def test_vfe_the_forced_zero_sentence_describes_a_running_standing_not_a_count_of_wins():
    """fix-explainer wrote H83's best slot as having 'won 62.5 of every 100 comparisons with its
    controls'. The number is a RUNNING AVERAGE of the share of its 8 controls the slot beat per
    window, seeded at one half (hierarchical_self.py): a slot that beat every control in all 3
    windows reads 63.6 in 100, so 62.5 is no count of comparisons won."""
    import inspect
    from mimo_bridge import hierarchical_self as HS
    src = inspect.getsource(HS)
    assert "self.rank_momentum = 0.9" in src and 'f"{tag}_rank_ema",' in src
    assert "torch.full((self.n_slots,), 0.5)" in src
    ceiling = 0.5
    for _ in range(3):
        ceiling = 0.9 * ceiling + 0.1 * 1.0
    assert f"{100 * ceiling:.1f}" == "63.6"
    w = BX.SCH.DOC["term.hself-measurements"]["w"]
    assert "won 62.5 of every 100 comparisons" not in w
    assert "starts at 50 and closes a tenth of the gap to 100" in w and "reads 63.6, not 100" in w
    # the inspector's hierarchical-self and entity-self rows print the same running average: as a
    # standing, never as comparisons won (entity_self.py keeps the same seeded average)
    from mimo_bridge import entity_self as ES
    assert 'f"{tag}_rank_ema", torch.full((N,), 0.5)' in inspect.getsource(ES)
    js = _viewer_js()
    assert not re.findall(r'"[^"\n]*comparisons won[^"\n]*"', js), "a page string calls the standing wins"
    assert '"a standing of " + Math.round(100 * v) + " in 100"' in js
    assert '"a standing of " + Math.round(100 * best) + " in 100"' in js


def test_vfe_the_senses_slide_writes_the_fresh_vision_share_as_a_count():
    """'the fresh-vision share reads 1.0' is a share printed as a decimal; bare_decimals passes
    it only because it is not below 1 (DESIGN.md section 11 asks for a count per 100)."""
    from mimo_bridge.explainer import walkthrough as WT
    steps = {st["id"]: st["body"] for st in WT.build_steps(
        {"active_modalities": SIX, "forward_model": {"_seq_length": 24, "_n_modalities": 6}}, {})}
    assert "fresh-vision share reads 100 of every 100 steps" in steps["senses"]
    assert "reads 1.0" not in steps["senses"]


def test_vfe_the_goal_rows_render_both_new_branches_without_a_bare_decimal(tmp_path, monkeypatch):
    """fix-explainer's goal rows have three branches: a goal the model RECORDED (followed through the
    ring), a snapshot whose model records none, and an older capture. Every real snapshot on disk
    is an older capture and no fixture carried the other two, so the page check had never drawn
    them. Drawn here on the build fixture: no page problem, no bare decimal, the rows present."""
    CK, exe = _browser_or_skip()
    run = _fixture_run_gof(tmp_path, monkeypatch)
    names = sorted(n for n in os.listdir(os.path.join(run, "explain")) if n.startswith("step_"))
    assert len(names) >= 2
    for k, name in enumerate(names[:2]):
        sp = os.path.join(run, "explain", name)
        snap = json.load(open(sp))
        it = snap.setdefault("intention", {})
        if not isinstance(it, dict):
            it = snap["intention"] = {}
        step = snap["header"]["step"]
        # a plan record shaped like segment 31's (explain/step_343139.json's intention.last)
        last = {"plan_length": 2, "horizon": 3, "converged": False, "final_mse": 0.8,
                "start_mse": 1.0, "mse_by_depth": [1.0, 0.8], "stopped_by": "horizon",
                "closer_than_start": True, "gap_closed_pct": 20.0, "goal_locality": None,
                "intention_execute_flag": True, "reached_body": None, "step": step - 3}
        it.update(last=dict(it.get("last") or {}, **last), goal_ring_size=200,
                  best_remembered_wellness=5.25, best_felt_value=4.69,
                  stored_value_ceiling_no_curiosity=7.35)
        if k == 0:
            it.update(goal_source="recorded", goal_plan_step=step - 3, goal_age_at_plan=40,
                      goal_age_now=43, goal_ring_index_now=156, goal_stored_value=5.1,
                      goal_felt_value=4.2, goal_curiosity_factor=1.21,
                      goal_wins_without_curiosity=False)
        else:
            it.update(goal_source="not recorded",
                      goal_index_absent="this run's model does not record which remembered model step "
                                        "its planner chose")
        json.dump(snap, open(sp, "w"))
    page = tmp_path / "goal_rows.html"
    page.write_text(BX.render(run), encoding="utf-8", newline="")
    seen = {}
    orig = CK.scan_rendered_texts

    def keep(texts):
        seen["t"] = " ".join(str(t[1]) for t in texts)
        return orig(texts)
    monkeypatch.setattr(CK, "scan_rendered_texts", keep)
    rep = CK.check_page(str(page), exe)
    assert rep["ran"] and not rep["problems"], rep.get("problems")
    n, hits = _page_bare_decimals(rep)
    assert n == 0, hits
    text = seen["t"]
    assert "43 model steps old at this snapshot" in text and "when it was 40 model steps old" in text
    assert "raised 21% by the curiosity drive" in text
    assert "not named: this run's model does not record which remembered model step" in text


def test_the_receipt_is_read_where_the_archive_moved_it_and_a_page_without_one_says_so(tmp_path, monkeypatch):
    """Review of 2026-09-25, finding 14. After tools/archive_checkpoints.py moved a segment's
    checkpoints to D:, the rebuilt page had no model receipt: 35 of its 52 boxes, no efference box,
    and nothing said why. The receipt is read where the move manifest says the checkpoint went, proven
    against the hash the manifest recorded; a copy that is not that file is refused; and a page with
    no receipt says so where the diagram starts."""
    import hashlib
    import shutil
    from mimo_bridge import storage_locations as SL
    run = _fixture_run(tmp_path / "moved")
    ck = os.path.join(run, "checkpoints", "ckpt_step_100")
    want = json.load(open(os.path.join(ck, "model_receipt.json"), encoding="utf-8"))
    dest = str(tmp_path / "archive" / "ckpt_step_100")
    shutil.move(ck, dest)
    files = {name: hashlib.sha256(open(os.path.join(dest, name), "rb").read()).hexdigest()
             for name in os.listdir(dest) if os.path.isfile(os.path.join(dest, name))}
    manifest = tmp_path / "move_manifest.jsonl"
    SL.append_row(manifest, {"event": "moved", "source": ck, "destination": dest, "files": files})
    monkeypatch.setattr(SL, "default_manifests", lambda: (manifest,))
    receipt, note = BX.find_receipt(run)
    assert receipt == want and dest in note and "checkpoint 100 there" in note, note
    page = BX.render(run)
    meta = _blocks(page)["data-meta"]
    assert meta["receipt"] == want and meta["receipt_note"] == note
    assert 'id="receipt-note"' in page
    # a copy that is not the file the manifest recorded is refused, and the page says it has none
    with open(os.path.join(dest, "model_receipt.json"), "a", encoding="utf-8") as fh:
        fh.write(" ")
    receipt, note = BX.find_receipt(run)
    assert receipt == {} and note.startswith("No model receipt") and "could not be proven" in note, note
    # no manifest names the run's checkpoints
    monkeypatch.setattr(SL, "default_manifests", lambda: ())
    receipt, note = BX.find_receipt(run)
    assert receipt == {} and "no storage manifest names a moved one" in note, note
    page = BX.render(run)
    assert '<p class="note" id="receipt-note">No model receipt' in page
    # a receipt in the run folder needs no note, and no manifest is read for it
    monkeypatch.setattr(SL, "default_manifests", lambda: (_ for _ in ()).throw(AssertionError("read")))
    assert BX.find_receipt(_fixture_run(tmp_path / "local"))[1] is None


def test_the_browser_check_fails_a_receipt_less_page_that_does_not_say_so(tmp_path, monkeypatch):
    """Review of 2026-09-25, finding 14: the check read 0 problems on a page with no receipt. A page
    that says it has none passes that line; the same page with its note removed fails it."""
    import shutil
    from mimo_bridge import storage_locations as SL
    CK, exe = _browser_or_skip()
    monkeypatch.setattr(SL, "default_manifests", lambda: ())
    run = _fixture_run(tmp_path)
    shutil.rmtree(os.path.join(run, "checkpoints", "ckpt_step_100"))
    page = BX.render(run)
    said = tmp_path / "no_receipt.html"
    said.write_text(page, encoding="utf-8", newline="")
    rep = CK.check_page(str(said), exe)
    assert rep["ran"] and rep["receipt_missing"] and "No model receipt" in (rep.get("receipt_note") or "")
    assert not any("no model receipt and does not say so" in p for p in rep["problems"]), rep["problems"]
    silent = tmp_path / "silent.html"
    silent.write_text(re.sub(r'<p class="note" id="receipt-note">.*?</p>', "", page, flags=re.S),
                      encoding="utf-8", newline="")
    rep = CK.check_page(str(silent), exe)
    assert any("no model receipt and does not say so" in p for p in rep["problems"]), rep["problems"]


def test_the_gaze_copy_popup_names_its_activation_repair(tmp_path):
    """Review of 2026-09-25, finding 11: production launches with --vision-activation-calibration,
    under which the gaze copy's hidden layer runs as a frozen copy's output plus the current weights'
    difference from it, through a GELU shifted and scaled per unit
    (lem/model/activation_compensation.py); the popup described only the plain two-layer path. It
    names the repair where the receipt records it, and says so where the launch names a calibration
    the receipt does not record."""
    import inspect
    from lem.model import activation_compensation as AC
    src = inspect.getsource(AC.VisualActivationCompensation.forward_with_preactivation).replace(" ", "")
    assert "returnprevious+delta" in src and "+self.shift" in src and "*self.gain" in src
    for name, receipt_extra, calibration, needle in (
            ("repaired", {"visual_activation_compensation": {"version": AC.VERSION}}, "calibration.json",
             "the gaze copy runs with the activation repair measured for it"),
            ("unrecorded", None, "calibration.json", "its model receipt records no repair"),
            ("plain", None, None, "runs the gaze copy's plain path")):
        run = _fixture_run_gaze_copy(tmp_path / name, "off")
        if calibration:
            _set_args(run, vision_activation_calibration=calibration)
        if receipt_extra:
            _edit_receipt(run, lambda r, extra=receipt_extra: r.update(extra))
        w = _blocks(BX.render(run))["data-doc"]["edge.efference.heads"]["w"]
        assert needle in w and "{gaze_copy_repair_run}" not in w, (name, w)


# the claims the efference popups carried before 2026-09-25, or that the reviewer planted and the
# suite let through (review of 2026-09-25, finding 7): none may come back into a popup or the viewer
_REMOVED_EFFERENCE_CLAIMS = ("has no production planner connection", "Production planning does not use",
                             "does not by itself change which command is chosen", "Drawn on every run with the heads",
                             "up to three commands ahead", "Under --wellness-rank raw both are among the values")


def test_the_efference_popups_hold_to_the_codes_numbers_and_refuse_the_removed_claims():
    """The trust region's numbers are the efference scales the search is built from
    (lem/model/physical_goal.py PhysicalTrustRegion reads EFFERENCE_SCALES pos_tenure_cm and
    rot_tenure_deg; the reviewer doubled them to 3.60 cm and the suite passed), the search plans one
    order, and no removed claim returns."""
    from lem.model.efference import EFFERENCE_SCALES as SC
    from lem.model import physical_goal as PG
    import inspect
    src = inspect.getsource(PG.PhysicalTrustRegion.__init__)
    assert "EFFERENCE_SCALES['pos_tenure_cm']" in src and "EFFERENCE_SCALES['rot_tenure_deg']" in src
    doc = _page_doc()
    pos, rot = SC["pos_tenure_cm"], SC["rot_tenure_deg"]
    w = doc["edge.efference.intention"]["w"]
    assert f"measured tenure spreads ({pos:.2f} cm and {rot:.1f} degrees per axis)" in w, w
    assert f"at most {pos:.2f} cm along each axis and turn it at most {rot:.1f} degrees" in w, w
    assert "the plan is one command long" in w
    js = _viewer_js()
    for key in ("node.efference", "edge.efference.heads", "edge.efference.wellness", "edge.efference.intention"):
        text = " ".join(json.dumps(doc[key]).split())
        for claim in _REMOVED_EFFERENCE_CLAIMS:
            assert claim not in text, (key, claim)
    for claim in _REMOVED_EFFERENCE_CLAIMS:
        assert claim not in js, claim


def test_a_page_whose_heads_lack_the_gaze_copy_draws_no_gaze_copy_line(tmp_path):
    """A model without vision builds the efference heads with no gaze copy, and its stamp says so
    (efference.py stamp: gaze_copy_into_vision = vision_idx is not None). No test rendered such a
    page, so the line drawn on every page with heads passed (the reviewer's plant: `if (true)` in
    place of the stamp's condition, review of 2026-09-25, finding 7)."""
    CK, exe = _browser_or_skip()
    run = _fixture_run_gaze_copy(tmp_path, "off")
    _edit_receipt(run, lambda r: r["efference_heads"].update(gaze_copy_into_vision=False))
    args = json.load(open(os.path.join(run, "args.json")))
    page_text = BX.render(run)
    b = _blocks(page_text)
    assert ("efference", "heads") not in _drawable_edges(b["data-meta"]["receipt"], args)
    slide = next(st for st in b["data-tour"] if st["id"] == "efference")
    assert ["efference", "heads"] not in slide["edges"]
    check = """<script>addEventListener('load',()=>{
      if(document.querySelector('.dg-edge[data-a="efference"][data-b="heads"]'))
        throw Error('the gaze copy line is drawn on a page whose stamp records no gaze copy');
      if(!document.querySelector('.dg-edge[data-a="motor"][data-b="efference"]'))
        throw Error('the efference box lost its command line: the page did not draw the heads at all');
    });</script>"""
    page = tmp_path / "no_gaze_copy.html"
    page.write_text(page_text.replace("</body>", check + "</body>"), encoding="utf-8", newline="")
    rep = CK.check_page(str(page), exe)
    assert rep["ran"], rep.get("why")
    assert not rep["problems"], "\n".join(rep["problems"])


def test_the_planner_popup_names_the_stored_value_and_the_plan_length_the_code_has(tmp_path):
    """Review of 2026-09-25, finding 10. The stored value's list left out the consequence term and the
    reader's value, the curiosity sentences did not say they need --curiosity-goal on, and the depth
    claim held for no plan of production's (--hand-target-mode wrist_relative_v1 makes every plan one
    order long). Each is held to the lines of lem_model.py that make it."""
    import inspect
    from mimo_bridge import lem_model as LM
    lsrc = open(inspect.getfile(LM), encoding="utf-8").read()
    chain = inspect.getsource(LM.LemLite._plan_chain)
    assert re.search(r"if self\.hand_target_mode != 'world':\s*\n\s*max_steps = 1", chain)
    assert "physical_cost = PhysicalGoalCost(" in chain and chain.count("max_steps = 1") == 2
    for term in ("wball = wball + self.consequence_weight * cq",
                 "wball = wball + self._self_value.value(self._entity_presence)",
                 "if self.curiosity_goal and self.curiosity_measured:",
                 "self.hand_response.choose_hand_goal("):
        assert term in lsrc, term
    doc = _page_doc()
    w = " ".join(doc["node.intention"]["w"].split())
    for needle in ("the agency payment under --agency-drive on", "the consequence term under a --consequence-w",
                   "the reader's value under --entity-self with a --self-value-weight above 0",
                   "under --curiosity-goal on the value is then"):
        assert needle in w, needle
    for name, extra, needles in (
            ("hands", {"intention": "on", "hand_target_mode": "wrist_relative_v1", "hand_goals": "memory"},
             ("every plan it makes is one command long", "set the finger targets by the second search")),
            ("plain", {"intention": "on", "hand_target_mode": "world", "hand_goals": "off"},
             ("plans run to their licensed depth", "the fingers' goals come from the search alone")),
            ("off", {"intention": "off"}, ("planner is off",))):
        run = _fixture_run(tmp_path / name)
        _set_args(run, **extra)
        h = _blocks(BX.render(run))["data-doc"]["node.intention"]["h"]
        assert all(n in h for n in needles) and "{plan_length_run}" not in h and "{hand_goals_run}" not in h, (
            name, h)


# ======================================================================================
# THE ARCHITECTURE TAB'S DECIDED DESIGN of Oct 7, 2026 (docs/H84_First_Movement_Plan_v3_2026-10-07/
# explainer_architecture_design.md, D-E1 to D-E7, after explainer_design_adversarial_review.md): the tour in the
# model diagram's own panel with slides and an end button (D-E1), the orange marks of the parts being worked on
# (D-E2), the violet tour (D-E3), the two repair paths inside their host boxes and the attention cap (D-E4), block 0
# at full size and blocks 1 to 3 as compact rows that keep their ids (D-E5), one size step larger text (D-E6). The
# header (D-E7) is held in test_explainer_refresh.py and the closing test in test_component_manifest.py.
# ======================================================================================
def _dom_probe(page_html, tmp_path, exe, probe, fragment=""):
    """The JSON object a probe returns after the page has booted, read from a headless browser's dump of a COPY of
    the page (--disable-gpu, a throwaway profile). `probe` is the body of a function run 300 ms after the load event;
    an exception inside it fails the test with its stack."""
    import html as _html
    import subprocess
    work = tmp_path / "dom_probe"
    work.mkdir(exist_ok=True)
    script = ("<script>addEventListener('load',()=>setTimeout(()=>{let R;try{R=(function(){" + probe + "})();}"
              "catch(e){R={probe_error:String((e&&e.stack)||e)};}const p=document.createElement('pre');"
              "p.id='probe-result';p.textContent=JSON.stringify(R);document.body.appendChild(p);},300));</script>")
    head, sep, tail = page_html.rpartition("</body>")
    page = work / "probed.html"
    page.write_text(head + script + sep + tail, encoding="utf-8", newline="")
    proc = subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                           "--disable-extensions", "--user-data-dir=" + str(work / "profile"),
                           "--virtual-time-budget=10000", "--window-size=1400,1000", "--dump-dom",
                           page.as_uri() + fragment], capture_output=True, timeout=300)
    found = re.findall(r'<pre id="probe-result">(.*?)</pre>', proc.stdout.decode("utf-8", "replace"), re.S)
    assert found, "the page wrote no probe result (exit code %s)" % proc.returncode
    out = json.loads(_html.unescape(found[-1]))
    assert "probe_error" not in out, out["probe_error"]
    return out


def _planted_component(cid, kind, blocks, states=(), tests=(), closing=None, note=None):
    """One component as component_manifest.status() writes it into the page, with open defects carrying `states`
    as this component's repair state (and `note` as the register's note on this repair), and `tests` as (registry
    row id, section) pairs."""
    return dict(id=cid, kind=kind, name=cid.lower().replace("_", " "), does="A planted component of the test.",
                blocks=list(blocks), flags=[], ref=None, confirmed={}, entry_fresh=True, entry_stale_text=None,
                last_code_change_text=None, last_qa_text=None, qa_stale=False, qa_stale_text=None, switch={},
                open_defects=[dict(id="DEFECT_OF_%s_%d" % (cid, i), title="A planted defect.", severity="GOF-BLOCKER",
                                   state=s, repair_state=s, repair_note=note, age_text="1 hour since it was found",
                                   idle_text=None,
                                   closing_test=closing if closing is not None else dict(row=None, text=None))
                              for i, s in enumerate(states)],
                linked_tests=[dict(id=row, title="a planted row", section=section, outcome=None)
                              for row, section in tests])


def test_the_orange_marks_follow_the_decided_rule_and_a_queued_row_marks_nothing(tmp_path):
    """D-E2, executed on a fixture page whose component view and switches are planted. The coordinator's controls: a
    planted BUILT/OFF repair gets a dashed ring, a planted IN TRAINING repair switched on in the page's run gets a solid
    ring, a QUEUED row marks nothing, and a FAILED/SUPERSEDED repair switched on gets a list line and no ring. Beside
    them: the same IN TRAINING state with the switch off gets a list line and no ring (the switch decides); a box
    drawing a solid and a dashed repair takes the solid ring; a RUNNING row rings the one box its part is drawn in and
    no box of a part drawn in two; the rings sit in their own layer right after the drawing's defs (behind every box);
    the list prints NAME: STATE, the switch, the boxes and each closing test, and holds back a closing text whose
    score is a bare decimal; the chips print the state's own word in the mark's style."""
    CK, exe = _browser_or_skip()
    run = _fixture_run(tmp_path)
    page = BX.render(run)
    meta = _blocks(page)["data-meta"]
    long_note = "A planted note longer than a line holds: " + "the branch, its review and what the state rests on; " * 6
    comps = [_planted_component("PLANT_BUILT", "repair", ["node.recon"], ["BUILT/OFF"],
                                closing=dict(row="planted_row", text="the planted closing text"), note=long_note),
             _planted_component("PLANT_TRAIN", "repair", ["node.gaze"], ["IN TRAINING"],
                                note="Switched on in the test; its own test showed the fault still there. Purpose not met."),
             _planted_component("PLANT_BUILT_TOO", "repair", ["node.gaze"], ["BUILT/OFF"],
                                closing=dict(row="r2", text="the token reads it at 0.25 or more")),
             _planted_component("PLANT_QUEUED", "model part", ["node.wellness"], tests=[("queued_row", "QUEUED")]),
             _planted_component("PLANT_FAILED", "repair", ["node.curiosity"], ["FAILED/SUPERSEDED"]),
             _planted_component("PLANT_OFF", "repair", ["node.intention"], ["IN TRAINING"]),
             _planted_component("PLANT_RUNNING", "model part", ["node.replay"], tests=[("running_row", "RUNNING")]),
             _planted_component("PLANT_RUNNING_WIDE", "model part", ["node.block_s1"], tests=[("wide_row", "RUNNING")])]
    by_block = {}
    for c in comps:
        for key in c["blocks"]:
            by_block.setdefault(key, []).append(c["id"])
    view = dict(read_at_text="Oct 7, 2026 9:00 PM PDT", no_qa_text="no independent QA on record", active_run=None,
                by_block=by_block, check=dict(ok=True, problems=[], sentence="The planted view."), components=comps,
                error=None)

    def switch(flag, value):
        return dict(flag=flag, value=value, present=value is not None, on=value == "on")
    meta["repair_switches"] = {"PLANT_BUILT": switch("--plant-built", None), "PLANT_TRAIN": switch("--plant-train", "on"),
                               "PLANT_BUILT_TOO": switch("--plant-built-too", "off"),
                               "PLANT_FAILED": switch("--plant-failed", "on"), "PLANT_OFF": switch("--plant-off", "off")}
    page = _replace_block(_replace_block(page, "data-components", view), "data-meta", meta)
    R = _dom_probe(page, tmp_path, exe, """
      const rings = {};
      document.querySelectorAll('#diagram rect.dg-ring').forEach(r => {
        rings[r.dataset.for] = r.classList.contains('solid') ? 'solid' : r.classList.contains('dashed') ? 'dashed' : '?'; });
      const lines = {};
      document.querySelectorAll('#worked-on li[data-component]').forEach(li => {
        lines[li.dataset.component] = {mark: li.dataset.mark, text: li.textContent}; });
      const chips = {};
      document.querySelectorAll('#components-index .comp-index-item').forEach(b => {
        const c = b.querySelector('.state-chip'); if (c) chips[b.dataset.component] = [c.textContent, c.className]; });
      const layer = document.querySelector('#diagram g.dg-rings');
      const folded = document.querySelector('#worked-on li[data-component="PLANT_BUILT"] details.worked-note');
      showComponent('PLANT_TRAIN');
      const card = [...document.querySelectorAll('#docpanel .comp-card .kv b')].map(b => b.textContent);
      const cardChip = document.querySelector('#docpanel .comp-card .comp-head .state-chip');
      return {rings, lines, chips, rule: (document.querySelector('#worked-on .worked-rule') || {}).textContent || '',
              behindTheBoxes: !!layer && !!layer.previousElementSibling && layer.previousElementSibling.tagName === 'defs',
              folded: folded ? [folded.open, folded.querySelector('summary').textContent] : null, card,
              cardChip: cardChip ? [cardChip.textContent, cardChip.className] : null,
              outlined: [...document.querySelectorAll('#diagram .dg-node.comp-hl')].map(g => g.id.slice(3))};
    """)
    assert R["rings"] == {"recon": "dashed", "gaze": "solid", "replay": "solid"}, R["rings"]
    assert R["behindTheBoxes"], "the ring layer is not drawn right after the defs, behind the boxes"
    assert {k: v["mark"] for k, v in R["lines"].items()} == {
        "PLANT_BUILT": "dashed", "PLANT_TRAIN": "solid", "PLANT_BUILT_TOO": "dashed", "PLANT_FAILED": "listed",
        "PLANT_OFF": "listed", "PLANT_RUNNING": "solid"}, R["lines"]
    lines = {k: " ".join(v["text"].split()) for k, v in R["lines"].items()}
    doc = _page_doc()
    assert "PLANT_TRAIN: IN TRAINING · switched on in this page's run, H99_fixture (--plant-train on)" in lines["PLANT_TRAIN"]
    assert "closing test not written" in lines["PLANT_TRAIN"]
    assert "PLANT_BUILT: BUILT/OFF · switched off in this page's run, H99_fixture (--plant-built absent)" in lines["PLANT_BUILT"]
    assert "drawn in: " + doc["node.recon"]["t"] in lines["PLANT_BUILT"]
    assert "closing test: registry row planted_row. the planted closing text" in lines["PLANT_BUILT"]
    withheld = "(1 sentence withheld here: it prints a score as a bare decimal"
    assert "0.25" not in lines["PLANT_BUILT_TOO"] and withheld in lines["PLANT_BUILT_TOO"], lines["PLANT_BUILT_TOO"]
    assert "PLANT_FAILED: FAILED/SUPERSEDED · switched on" in lines["PLANT_FAILED"]
    assert "PLANT_OFF: IN TRAINING · switched off" in lines["PLANT_OFF"]
    assert "registry row running_row is RUNNING" in lines["PLANT_RUNNING"]
    assert R["chips"] == {"PLANT_BUILT": ["BUILT/OFF", "state-chip dashed"], "PLANT_TRAIN": ["IN TRAINING", "state-chip solid"],
                          "PLANT_BUILT_TOO": ["BUILT/OFF", "state-chip dashed"],
                          "PLANT_FAILED": ["FAILED/SUPERSEDED", "state-chip plain"],
                          "PLANT_OFF": ["IN TRAINING", "state-chip plain"],
                          "PLANT_RUNNING": ["RUNNING", "state-chip solid"]}, R["chips"]
    for words in ("solid ring", "IN TRAINING", "dashed ring", "BUILT/OFF, REVIEWED or QUALIFIED/OFF",
                  "QUEUED marks nothing", "Oct 7, 2026 9:00 PM PDT"):
        assert words in R["rule"], words
    # the register's note on the repair: a short one in the line, a long one folded under it
    assert "the register's note on this repair: Switched on in the test; its own test showed the fault still there. " \
           "Purpose not met." in lines["PLANT_TRAIN"], lines["PLANT_TRAIN"]
    assert R["folded"] == [False, "the register's note on this repair"], R["folded"]
    # a line opens its component's card (with the same chip) and outlines its boxes; the card's row names the check
    # of the entry against the code, not its result (the coordinator's change of Oct 7, 2026: on 51 of 81 cards the
    # value reads "carried over from the explainer's text, not checked against the code")
    assert R["cardChip"] == ["IN TRAINING", "state-chip solid"] and R["outlined"] == ["gaze"], (R["cardChip"], R["outlined"])
    assert "check of this entry against the code" in R["card"] and "entry checked against the code" not in R["card"], R["card"]


# The defect register's texts on VESTIBULAR_TOKEN_NO_HEAD_TILT as they read on Oct 15, 2026 (docs/gain_of_function_v4/
# defect_register.json): HEAD_TILT_REPAIR's note and the defect's closing test. viewer.js WORK_PLAIN_REWRITES holds the
# plain rewrite of each sentence here that prints a bare decimal.
HEAD_TILT_NOTE = (
    "Built on branch fix/vestibular_token_readouts-2026-10-03 (fe276c7de, path version 2); merged into production off "
    "by default at 1bd91c3b1 (Oct 8, 2026, 2:52 AM PDT) through merge/head-tilt-repair-2026-10-07 (865ece161, which "
    "also holds the text corrections b30453b31). The module file is the same blob at fe276c7de and at HEAD. Independent "
    "reviews on record: fe276c7de (review/head-tilt-repair-2026-10-05, Oct 5: deploy with named changes; the watch "
    "text, the segment row's wiring names and the fixed-vector statement corrected on Oct 5; its whole-model check ran) "
    "and the merge 865ece161 (review/head-tilt-merge-2026-10-08, Oct 8: MERGE AFTER NAMED FIXES; open: the watch "
    "header's wiring text, needed only for on_delayed_encoder_only, and the relaunch's preflight record). Since "
    "865ece161 eight wiring definitions changed by lines that compute nothing of the path "
    "(D:/UEMA_Research/external/D/UEMA_scratch/head_tilt_reconciliation_20261013/RECONCILIATION.md, section 4). Its whole-model check ran on "
    "checkpoint 481,800 (Oct 5): head tilt in the token at 0.44 to 0.50 (line 0.25), vision's cost above the seed "
    "line. Two further runs of on did not repeat the vision cost: seed 1 (Oct 5) and seed 0 with the first run's "
    "replay draws (Oct 8, PASS: median ln ratio 0.0089 against 0.0342 read, 0.0069 against 0.0195 unread). By the "
    "operator's rule of Oct 7 that PASS selected the wiring on for the restart; the operator's decision of Oct 8 "
    "stopped automatic deployment, so the switch stays off.")
HEAD_TILT_CLOSING = (
    "With the path on in production, the trainer's own probe reads each body's change of head tilt from the vestibular "
    "token at 0.25 or more in every window of 1,000 model steps from 2,001 model steps after the switch, with no stop "
    "line crossed through 10,000 model steps.")
# PLANNER_SEARCH_UNSTABLE's closing test as it reads on Oct 15, 2026
SEARCH_CLOSING = (
    "The search selects nearly the same command from nearly the same input (same-state repeatability and the 0.00001 "
    "perturbation), and the sense weights are then tested again with a paired gate (checklist item 4g). The row for "
    "the repaired search is the planner search agent's to write.")


def test_the_work_list_keeps_a_notes_decision_failure_result_and_pass_line_sentence_by_sentence(tmp_path):
    """The independent review of Oct 15, 2026, finding M2, and the terms of that day (explanations-fence-content-
    followup-20261015): a note or a closing text holding one bare decimal was replaced WHOLE, which hid the operator's
    decision of Oct 8, a failed measurement and the head-tilt closing test's pass line. Now each sentence is judged
    alone: the head-tilt note keeps its decision sentence as written, its failed check and its later PASS print as plain
    rewrites with the same values on the scales the registry rows state, the closing test's pass line prints as its
    rewrite, a planted unexplained sentence is withheld alone with its count, and a quotation holding a full stop stays
    whole with its numbers."""
    CK, exe = _browser_or_skip()
    run = _fixture_run(tmp_path)
    page = BX.render(run)
    meta = _blocks(page)["data-meta"]
    tilt = _planted_component("HEAD_TILT_REPAIR", "repair", ["node.recon"], ["BUILT/OFF"],
                              closing=dict(row="vestibular_input_path_production_segment", text=HEAD_TILT_CLOSING),
                              note=HEAD_TILT_NOTE)
    tilt["open_defects"][0]["id"] = "VESTIBULAR_TOKEN_NO_HEAD_TILT"
    mixed = _planted_component("PLANT_MIXED", "repair", ["node.gaze"], ["BUILT/OFF"],
                               note=('First sentence, kept with 3 of 8 bodies. The token reads it at 0.25 or more. '
                                     'The review said "deploy. With 2 named changes." Last sentence, kept.'))
    # the second recheck of Oct 15: "the 0.00001 perturbation" passes the unit list ("per" matches the start of
    # "perturbation"), so the table's rewrite is applied before the rule is asked
    search = _planted_component("SEARCH_REPAIR", "repair", ["node.intention"], ["BUILT/OFF"],
                                closing=dict(row=None, text=SEARCH_CLOSING))
    search["open_defects"][0]["id"] = "PLANNER_SEARCH_UNSTABLE"
    comps = [tilt, mixed, search]
    view = dict(read_at_text="Oct 15, 2026 11:30 AM PDT", no_qa_text="no independent QA on record", active_run=None,
                by_block={"node.recon": ["HEAD_TILT_REPAIR"], "node.gaze": ["PLANT_MIXED"],
                          "node.intention": ["SEARCH_REPAIR"]},
                check=dict(ok=True, problems=[], sentence="The planted view."), components=comps, error=None)
    meta["repair_switches"] = {"HEAD_TILT_REPAIR": dict(flag="--vestibular-input-path", value="off", present=True,
                                                        on=False),
                               "PLANT_MIXED": dict(flag="--plant-mixed", value="off", present=True, on=False),
                               "SEARCH_REPAIR": dict(flag="--plant-search", value="off", present=True, on=False)}
    page = _replace_block(_replace_block(page, "data-components", view), "data-meta", meta)
    R = _dom_probe(page, tmp_path, exe, """
      const lines = {};
      document.querySelectorAll('#worked-on li[data-component]').forEach(li => { lines[li.dataset.component] = li.textContent; });
      return {lines};
    """)
    tilt_line = " ".join(R["lines"]["HEAD_TILT_REPAIR"].split())
    mixed_line = " ".join(R["lines"]["PLANT_MIXED"].split())
    # the decision, as the register writes it
    assert ("By the operator's rule of Oct 7 that PASS selected the wiring on for the restart; the operator's decision "
            "of Oct 8 stopped automatic deployment, so the switch stays off.") in tilt_line, tilt_line
    # the failed check and the later PASS, with the same values (0.44 to 0.50 of 1 = 44 to 50 in 100; ln ratios 0.0089,
    # 0.0342, 0.0069, 0.0195 = 89, 342, 69, 195 in 10,000)
    assert "head tilt read from the token at 44 to 50 in 100" in tilt_line and "line 25 in 100" in tilt_line
    assert "vision's cost above the seed line, so that check's verdict was fail" in tilt_line
    assert ("Oct 8, PASS: the median of ln(vision's error ratio in the arm with the switch / in the arm without it), "
            "where 0 = the same error and 1,000 in 10,000 = an error 10.5 in 100 above the arm without the switch, at 89 "
            "in 10,000 against the line 342 in 10,000 on the read set and 69 in 10,000 against 195 in 10,000 on the "
            "unread set") in tilt_line
    # the closing test's pass line
    assert ("from the vestibular token at 25 in 100 or more (R squared of the probe's read-out: 0 = no better than "
            "predicting no change, 100 = every scored change predicted) in every window of 1,000 model steps") in tilt_line
    # the rest of the note as written, the notices with their counts, and no bare decimal left
    assert "Built on branch fix/vestibular_token_readouts-2026-10-03 (fe276c7de, path version 2)" in tilt_line
    assert "(2 sentences rewritten here in plain numbers on Oct 15, 2026" in tilt_line
    assert "(1 sentence rewritten here in plain numbers on Oct 15, 2026" in tilt_line
    assert "withheld here" not in tilt_line and "does not print" not in tilt_line
    for decimal in ("0.25", "0.44", "0.50", "0.0089", "0.0342"):
        assert decimal not in tilt_line, decimal
    # an unexplained sentence is withheld alone; its neighbours and a quotation with a full stop inside stay whole
    assert "First sentence, kept with 3 of 8 bodies." in mixed_line and "Last sentence, kept." in mixed_line
    assert 'The review said "deploy. With 2 named changes."' in mixed_line
    assert "0.25" not in mixed_line and "(1 sentence withheld here: it prints a score as a bare decimal" in mixed_line
    # the perturbation sentence prints as its rewrite, in counts, with the row it was read from; the next sentence as
    # written
    search_line = " ".join(R["lines"]["SEARCH_REPAIR"].split())
    assert "0.00001" not in search_line, search_line
    assert ("Gaussian noise of standard deviation 1 in 100,000 added to each of the 192 numbers of each of the five "
            "senses' present tokens") in search_line and "planner_weights_review_search_noise_479350" in search_line
    assert "The row for the repaired search is the planner search agent's to write." in search_line
    assert "(1 sentence rewritten here in plain numbers on Oct 15, 2026" in search_line


def _plain_rewrites_of_the_viewer():
    """viewer.js WORK_PLAIN_REWRITES as {key: rewrite}, read from the file's own text: the object literal is parsed
    with esprima and each key and value, a sum of string literals, is added up. A missing esprima FAILS the test, never
    skips it (the reviewer's P2C-M1; "Required live-register coverage cannot count a missing-parser skip as
    PASS")."""
    try:
        import esprima
    except ImportError as exc:
        pytest.fail("esprima, the parser that reads WORK_PLAIN_REWRITES out of viewer.js, is not installed: %s" % exc)
    src = _viewer_js()
    start = src.index("const WORK_PLAIN_REWRITES = {")
    end = src.index("\n};", start) + 3
    tree = esprima.parseScript(src[start:end])

    def text(node):
        if node.type == "Literal" and isinstance(node.value, str):
            return node.value
        assert node.type == "BinaryExpression" and node.operator == "+", node.type
        return text(node.left) + text(node.right)
    table = tree.body[0].declarations[0].init
    assert table.type == "ObjectExpression"
    out = {}
    for prop in table.properties:
        key = text(prop.key.elements[0]) if prop.key.type == "ArrayExpression" else text(prop.key)
        out[key] = text(prop.value)
    return out


def _register_sentences(text):
    """viewer.js registerSentences, ported: sentences end at a full stop followed by white space, outside parentheses
    and quotation marks."""
    s, out, depth, quote, start = str(text), [], 0, None, 0
    for i, ch in enumerate(s):
        if quote:
            if (quote == '"' and ch == '"') or (quote == "“" and ch == "”"):
                quote = None
            continue
        if ch in ('"', "“"):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "." and depth == 0 and i + 1 < len(s) and s[i + 1].isspace():
            out.append(s[start:i + 1].strip())
            start = i + 1
    if s[start:].strip():
        out.append(s[start:].strip())
    return out


def test_every_plain_rewrite_key_is_still_a_sentence_of_the_live_register():
    """The second recheck of Oct 15, 2026 (P2-m3), as tests/test_elimination.py holds for the registry's rewrites: each
    key of viewer.js WORK_PLAIN_REWRITES (defect | field | sentence) names a sentence that is still a sentence of that
    field in this checkout's defect register (the field "closing test" is the defect's closing_test.text; any other is
    the note of the defect's repair entry whose component it names, as component_manifest.status hands the page
    repair_note), split as the page splits it. A register edit would otherwise drop the rewrite without notice. Each
    rewrite prints no bare decimal under the whole-word unit rule."""
    from mimo_bridge.tools import defect_register as DR
    from mimo_bridge.tools.check_no_bare_decimals import bare_decimals
    table = _plain_rewrites_of_the_viewer()
    assert len(table) >= 4, sorted(table)
    defects = {d.get("id"): d for d in DR.load()["defects"] if isinstance(d, dict)}
    for key, plain in table.items():
        defect_id, field, sentence = key.split("|", 2)
        assert defect_id in defects, key[:80]
        d = defects[defect_id]
        if field == "closing test":
            source = (d.get("closing_test") or {}).get("text") or ""
        else:
            source = next((r.get("note") for r in d.get("repairs") or []
                           if isinstance(r, dict) and r.get("component") == field and isinstance(r.get("note"), str)),
                          "")
        sentences = [" ".join(x.split()) for x in _register_sentences(source)]
        assert sentence in sentences, ("the register no longer holds this sentence", key[:120])
        assert bare_decimals(plain) == [], (key[:80], plain)


def test_the_viewers_unit_list_is_the_scanners():
    """viewer.js hasBareDecimal ports tools/check_no_bare_decimals.bare_decimals: its one unit list S11_UNIT_STRICT is
    UNIT_AFTER_STRICT character for character, the scanner's active UNIT_AFTER is that list (switched on Oct 15, 2026,
    the decision explainer-part2-rule-confirmed-20261015), and hasBareDecimal reads it with no second argument."""
    from mimo_bridge.tools import check_no_bare_decimals as C
    src = _viewer_js()
    m = re.search(r"^const S11_UNIT_STRICT = /(.*)/i;$", src, re.M)
    assert m and m.group(1) == C.UNIT_AFTER_STRICT.pattern and C.UNIT_AFTER_STRICT.flags & re.I
    assert C.UNIT_AFTER is C.UNIT_AFTER_STRICT
    assert re.search(r"^const S11_UNIT = ", src, re.M) is None, "the word-prefix list is still in viewer.js"
    assert "function hasBareDecimal(text) {\n  const unit = S11_UNIT_STRICT;" in src


def test_the_tour_is_in_the_model_diagrams_panel_counts_slides_and_ends(tmp_path):
    """D-E1 and D-E3. The skeleton: the learning circle, then the model diagram's panel holding its heading, the tour
    and the drawing, then the detail panel. Opened at #tr=2 the tour shows "slide 3 of N" with N the tour's own
    length, previous, next and "end the tour"; the slide's boxes take the violet fill and an outline each, drawn as
    elements of their own in a layer after every box; its lines move. "end the tour" clears the fill, the outlines,
    the moving lines and the tr= part of the address."""
    CK, exe = _browser_or_skip()
    run = _fixture_run(tmp_path)
    page = BX.render(run)
    order = [page.index(s) for s in ('<div id="learnloop">', '<div id="model-diagram">', '<div id="tour">',
                                     '<div id="diagram">', '<div id="detail"')]
    assert order == sorted(order), "the tour is not between the learning circle and the drawing, in the drawing's panel"
    b = _blocks(page)
    tour, receipt = b["data-tour"], b["data-meta"]["receipt"]
    lit = sorted(set(tour[2]["nodes"]) & _drawable_nodes(receipt, json.load(open(os.path.join(run, "args.json")))))
    R = _dom_probe(page, tmp_path, exe, """
      const panel = document.getElementById('model-diagram'), svg = document.querySelector('#diagram svg');
      const marks = [...document.querySelectorAll('#diagram rect.dg-tour-mark')];
      const out = {heading: (panel.querySelector('h3') || {}).textContent || '',
                   inPanel: panel.contains(document.getElementById('tour')) && panel.contains(svg),
                   pos: document.querySelector('#tour .tour-pos').textContent,
                   buttons: [...document.querySelectorAll('#tour button')].map(x => x.textContent),
                   lit: [...document.querySelectorAll('#diagram .dg-node.hl')].map(g => g.id.slice(3)).sort(),
                   marks: marks.map(r => r.dataset.for).sort(),
                   marksOnTop: marks.length > 0 && marks[0].parentNode === svg.lastElementChild,
                   markStroke: marks.length ? getComputedStyle(marks[0]).stroke : null,
                   flows: document.querySelectorAll('#diagram .dg-edge.flow').length, hash: location.hash};
      [...document.querySelectorAll('#tour button')].find(x => x.textContent === 'end the tour').click();
      out.after = {lit: document.querySelectorAll('#diagram .dg-node.hl').length,
                   marks: document.querySelectorAll('#diagram rect.dg-tour-mark').length,
                   flows: document.querySelectorAll('#diagram .dg-edge.flow').length, hash: location.hash,
                   pos: document.querySelector('#tour .tour-pos').textContent,
                   buttons: [...document.querySelectorAll('#tour button')].map(x => x.textContent)};
      return out;
    """, fragment="#tr=2")
    n = len(tour)
    assert R["heading"].startswith("The model diagram: ") and R["inPanel"], R
    assert R["pos"].startswith("slide 3 of %d" % n), R["pos"]
    assert R["buttons"] == ["◀ previous", "next ▶", "end the tour"], R["buttons"]
    assert R["lit"] == lit and R["marks"] == lit, (R["lit"], R["marks"], lit)
    assert R["marksOnTop"] and R["markStroke"] == "rgb(107, 63, 160)", (R["marksOnTop"], R["markStroke"])
    assert R["flows"] > 0 and "tr=2" in R["hash"], R
    after = R["after"]
    assert after["lit"] == after["marks"] == after["flows"] == 0, after
    assert "tr=" not in after["hash"] and after["buttons"] == ["start the tour ▶"], after
    assert "%d slides" % n in after["pos"], after["pos"]
    # the tour's own words: slides, never "steps" (the style guide reserves "model step" for the 50 ms update)
    js = _viewer_js()
    tour_js = js[js.index("function renderTour()"):js.index("function snapByStep(")]
    assert "steps - click any box" not in tour_js and "`step ${" not in tour_js


def test_the_core_keeps_four_addressable_blocks_and_the_repair_paths_sit_in_their_host_boxes(tmp_path):
    """D-E4, D-E5, D-E6 on a 4-block fixture whose receipt carries the two repair paths and the liveness repair and
    whose launch had the attention cap on with bound 3. All 16 block part ids are drawn: block 0's as full boxes, the
    other 12 as cells; the frames carry the register's block names; the blocks are joined in series and the cells of a
    row by their links. The motor encoder's box carries "+ asked-change path, 132k" and the cap's bound, the stage-1
    box of block 0 the bound, the prediction heads' box "+ all-slot path, 403k"; the not-drawn list drops the two
    paths and keeps the liveness repair; the path's line opens the path's own entry. Box titles render at 12 units."""
    CK, exe = _browser_or_skip()
    run, _core = _fixture_run_at_shape(tmp_path, d_model=192, n_blocks=4)
    _edit_receipt(run, lambda rec: rec["component_params"].update(
        motor_asked_change=131728, heads_all_slot_path=403200, liveness_repair=2014180))
    _set_args(run, attention_cap="on", attention_cap_bound=3.0, motor_asked_change="on", heads_all_slot_path="on")
    page = BX.render(run)
    args = _blocks(page)["data-meta"]["args"]
    assert (args["attention_cap"], args["attention_cap_bound"], args["heads_all_slot_path"]) == ("on", 3.0, "on")
    R = _dom_probe(page, tmp_path, exe, """
      const txt = id => { const g = document.getElementById('nd-' + id);
        return g ? [...g.querySelectorAll('text')].map(t => t.textContent) : null; };
      const ids = []; for (let i = 0; i < 4; i++) for (const s of ['s1', 's2', 's3', 'ffn']) ids.push('block' + i + '_' + s);
      const edge = (a, b) => !!document.querySelector('.dg-edge[data-a="' + a + '"][data-b="' + b + '"]');
      const foot = [...document.querySelectorAll('#diagram svg > text')].map(t => t.textContent)
        .find(t => t.indexOf('not drawn: ') === 0) || '';
      const title = document.querySelector('#nd-heads > text');
      const hosted = document.querySelector('#nd-enc_motor text.hosted');
      if (hosted) hosted.dispatchEvent(new MouseEvent('click', {bubbles: true}));
      return {missing: ids.filter(i => !document.getElementById('nd-' + i)),
              cells: ids.filter(i => document.querySelector('#nd-' + i + ' text.cell')),
              frames: [...document.querySelectorAll('#diagram rect.dg-frame')].map(f => f.dataset.name),
              motor: txt('enc_motor'), heads: txt('heads'), s1: txt('block0_s1'), cell: txt('block1_s1'), foot,
              series: [edge('block0_ffn', 'block1_s1'), edge('block1_ffn', 'block2_s1'), edge('block2_ffn', 'block3_s1'),
                       edge('block3_ffn', 'final_norm')],
              row: [edge('block2_s1', 'block2_s2'), edge('block2_s2', 'block2_s3'), edge('block2_s3', 'block2_ffn')],
              titleSize: title ? getComputedStyle(title).fontSize : null,
              popup: (document.querySelector('#docpanel h4') || {}).textContent || null};
    """)
    assert R["missing"] == [], R["missing"]
    assert R["cells"] == ["block%d_%s" % (i, s) for i in (1, 2, 3) for s in ("s1", "s2", "s3", "ffn")], R["cells"]
    assert R["frames"] == ["the frame of block 0 (1st of 4)", "the frame of block 1 (2nd of 4)",
                           "the frame of block 2 (3rd of 4)", "the frame of block 3 (4th of 4)"], R["frames"]
    assert "+ asked-change path, 132k" in R["motor"] and "attention cap on, bound 3" in R["motor"], R["motor"]
    assert "attention cap on, bound 3" in R["s1"] and R["cell"] == ["stage 1"], (R["s1"], R["cell"])
    assert "+ all-slot path, 403k" in R["heads"], R["heads"]
    assert "liveness repair 2.01M" in R["foot"], R["foot"]
    assert "asked change" not in R["foot"] and "all slot path" not in R["foot"], R["foot"]
    assert all(R["series"]) and all(R["row"]), (R["series"], R["row"])
    assert R["titleSize"] == "12px", R["titleSize"]
    assert R["popup"] == _page_doc()["path.motor_asked_change"]["t"], R["popup"]


def test_every_gated_part_the_diagram_reads_has_a_switch_and_every_switch_is_read():
    """The model diagram's switches (operator, Oct 15, 2026: "Things that are behind gates should have a switch to show
    them on/off with the default being whatever is current in the code"). buildDiagram reads each gated part through
    gate("id", this run's value), and the panel lists GATES; a gate read with no row would have no switch, and a row
    no gate() call reads would switch nothing."""
    js = open(os.path.join(BX.PKG, "viewer.js"), encoding="utf-8").read()
    diagram = js[js.index("function buildDiagram()"):js.index("function buildLearnLoop()")]
    read = set(re.findall(r'\bgate\("([a-z_]+)"', diagram))
    listed = set(re.findall(r'^  \["([a-z_]+)", "', js[js.index("const GATES = ["):js.index("const GATE_SET")], re.M))
    assert read, "buildDiagram reads no gate - the check stopped finding gate() calls"
    assert read == listed, (f"gates read with no switch: {sorted(read - listed)}; "
                            f"switches no gate reads: {sorted(listed - read)}")
