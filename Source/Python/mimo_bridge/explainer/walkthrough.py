# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The architecture tour: ordered steps, each naming diagram nodes and explaining
one mechanism in real terms (encoder, loss, transformer, wellness - one-line plain
definition on first use, never a stand-in name).

Every number a RUN can change is read from that run's own model receipt or args at
build time - this function is handed the whole args dict, so a new one needs no new
plumbing. Numbers fixed in the model's own code and not reachable by any flag (the
goal ring's 200 model steps and the planner's 20 Adam rounds, both in lem_model.py) are
written here as those constants and named as fixed; changing them in code without
changing this file would make the tour wrong.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .schema import SENSE_DEFINITIONS, efference_routes, model_shape, moment_clock, vision_reach

# {layers} and {reach} in the Swin line are filled in build_steps from schema.vision_reach:
# the reach follows the layer count (10 places at 2 layers, 17 at 4), and the layer count is
# a launch setting, so neither may be typed here
ENCODER_KIND = {
    "swin_patch": "a Swin transformer - an image encoder whose attention works in "
                  "windows of 7 patches cut from the patch LIST, so at this run's "
                  "{layers} one patch reaches another only within {reach} places of it",
    "cross_patch_attention": "cross-patch attention - each contact/handle row attends "
                             "to the others, then one learned query pools them",
    "cross_handle_attention": "cross-handle attention - each of the body's handles "
                              "attends to the others, then one learned query pools them",
    "conv1d_stack": "a stack of 1-D convolutions over the spectrum",
    "mlp_project": "a multi-layer perceptron (stacked fully-connected layers)",
    "self_attention_pool": "self-attention over the agent slots, pooled by a "
                           "learned query",
}


def build_steps(receipt: Dict[str, Any], args: Dict[str, Any]) -> List[Dict[str, Any]]:
    fm = receipt.get("forward_model", {})
    mods = receipt.get("active_modalities", [])
    n_mod = int(fm.get("_n_modalities", len(mods) or 6))
    seq = int(fm.get("_seq_length", 0))
    ctx = seq // n_mod if n_mod else 0
    # ONE source for the model's size: schema.model_shape, which viewer.js reads as
    # META.shape - so the stage slides describe exactly the blocks the picture draws
    shape = model_shape(receipt, args)
    n_layers = shape["core_blocks"]
    n_heads = int(fm.get("n_heads", 0))
    handles = list(receipt.get("handle_names") or [])
    n_handles = len(handles) or 24
    per_handle = int((receipt.get("motor_controller") or {}).get("output_per_handle", 9) or 9)
    gaze_idx = handles.index("gaze") if "gaze" in handles else 23
    d_model = shape["d_model"]
    total = int(receipt.get("total_params", 0))
    hold = int(args.get("motor_hold", 5) or 5)
    _every = int(args.get("intention_every", 0) or 0)
    smooth = float(args.get("smooth_actions", 0.0) or 0.0)
    replay_every = int(args.get("replay_every", 4) or 4)
    replay_batch = int(args.get("replay_batch", 48) or 48)
    residual = args.get("residual_prediction") == "on"
    # ONE source for the model step's length: the receipt's measured model step (schema.moment_clock)
    step_s = moment_clock(receipt, args)["model_step_s"]
    # the chain's depth and lesson weight are read where the imagination slide is
    # built, and only when the launch actually recorded them - a default typed here
    # (3 and 0.5) would print a number the run never set as if it had
    # ONE gate flag and ONE gate route. Two names for each (td_on/gate_on,
    # gate_chain/norm_chain) arrived from two different fixes and were free to drift.
    # viewer.js routes the encoders' line through the top-down gate box when the gate
    # is on, so the slides' own lines have to follow the same route or they animate
    # nothing (the diagram draws ln_out -> assembly directly only when it is off)
    td_on = (args.get("top_down_gate") or "off") != "off"
    gate_chain = ([["ln_out", "td_gate"], ["td_gate", "assembly"]] if td_on
                  else [["ln_out", "assembly"]])
    audio_real = args.get("audio") == "real"
    caregiver = (args.get("caregiver_mode") or "off") != "off"
    voice_on = bool((receipt.get("voice_head") or {}).get("enabled"))
    mem_on = (receipt.get("persistent_memory") or args.get("persistent_memory")) == "on"
    self_mode = receipt.get("self_state") or args.get("self_state") or "off"
    hself_on = (args.get("hierarchical_self") or "off") != "off"
    eval_only = bool(args.get("eval_only"))
    vreach = vision_reach(receipt, args)
    kinds = dict(ENCODER_KIND)
    kinds["swin_patch"] = ENCODER_KIND["swin_patch"].format(
        layers=f"{vreach['layers']} layer" + ("" if vreach["layers"] == 1 else "s"),
        reach=vreach["reach"])
    enc_lines = "; ".join(
        f"{m}: {kinds.get(e.get('type'), e.get('type'))}"
        for m, e in receipt.get("encoders_built", {}).items())

    s: List[Dict[str, Any]] = []

    def step(sid, title, nodes, body, edges=None):
        # edges: the [from, to] diagram lines this slide animates. The viewer
        # marches EXACTLY these - never "every line touching a lit box", which
        # once lit side paths and left the loop's own middle dark (2026-09-14).
        s.append({"id": sid, "title": title, "nodes": nodes, "body": body,
                  "edges": edges or []})

    # the drawn spine of the core, in order: assembly -> stages -> final norm
    core_chain: List[List[str]] = [["assembly", "block0_s1"]]
    for i in range(n_layers):
        core_chain += [[f"block{i}_s1", f"block{i}_s2"],
                       [f"block{i}_s2", f"block{i}_s3"],
                       [f"block{i}_s3", f"block{i}_ffn"]]
        core_chain.append([f"block{i}_ffn", f"block{i + 1}_s1"] if i + 1 < n_layers
                          else [f"block{i}_ffn", "final_norm"])

    step("loop", "One loop, no dataset", ["world", "assembly", "block0_s1", "heads", "motor"],
         ("This page was built from an EVAL-ONLY probe: not one weight changed "
          "while it ran, and every training sentence in this tour describes what "
          "the model does when training is switched on. " if eval_only else "")
         + f"This model ({total:,} parameters in the optimizer at the size its launch "
         f"flags set: {d_model}-number tokens, {n_layers} blocks) lives inside a "
         f"closed loop: the body's senses are encoded into tokens, a transformer (an "
         f"attention-based sequence model) predicts what comes next and proposes "
         f"actions, the body executes them, and the next model step's senses grade the "
         f"predictions. Every part trains while the body lives - there is no dataset, "
         f"only the stream. Each model step covers "
         f"{step_s:.2f} s of body time. "
         f"The moving lines trace this whole circuit - world to senses to the core "
         f"to the outputs and back to the world.",
         edges=([["world", f"sense_{m}"] for m in mods]
                + [[f"sense_{m}", f"enc_{m}"] for m in mods]
                + [[f"enc_{m}", "ln_out"] for m in mods]
                + gate_chain + core_chain
                + [["final_norm", "heads"], ["final_norm", "motor"],
                   ["motor", "world"]]))

    step("senses", "The senses and their source labels",
         [f"sense_{m}" for m in mods],
         f"Active senses: {', '.join(mods)} - the model's own names for them, each defined here: "
         + "; ".join(f"{m} is {SENSE_DEFINITIONS[m]}" for m in mods if m in SENSE_DEFINITIONS)
         + f". Every frame each sense is labeled "
         f"real (measured this step), stale (the previous value re-used), or "
         f"synthetic (a generated placeholder). The loss "
         f"scales by this label, so a placeholder never teaches. Nothing goes "
         f"stale in these runs: the trainer renders on the LAST sub-step of every "
         f"model step, so every model step gets a newly rendered picture - the "
         f"run tab's fresh-vision share reads 100 of every 100 steps when that holds, "
         f"and it reads 100 of every 100 for the whole of this run. "
         + ("Audio is REAL here: the caregiver's speech, body impacts, the "
            "squeaker, the mobile's chime and the flying critters reach two "
            "simulated ears as a 256-band spectrum each. Direction is NOT inside "
            "those 512 numbers: every synthesized source hands both ears the same "
            "spectral shape at two loudnesses (its own distance to each ear, and a "
            "head shadow that leaves the far ear at 65% of the near ear's level), with no phase and no per-bin "
            "difference. The arrival-time difference travels beside the spectrum "
            "as one number in milliseconds, at most 0.204 ms - the 7 cm between the "
            "ears divided by 343 m/s. Only the caregiver's recorded speech carries "
            "a true per-ear delay in its waveform. "
            "The model cannot hear its own voice yet - nothing carries the "
            "voice head's output into the world. "
            if audio_real else
            "Audio has no source in this world, so it is a synthetic placeholder. ")
         + ("Intermind stays synthetic (nothing in this world produces it)."
            if "intermind" in mods else "")
         + ("The sixth slot carries vestibular (--vestibular-sense on): the "
            "body's balance, neck, support and effort as 29 numbers, where the intermind "
            "placeholder's noise used to go." if "vestibular" in mods else ""),
         edges=[["world", f"sense_{m}"] for m in mods])

    step("encoders", "One encoder per sense, one shared scale",
         [f"enc_{m}" for m in mods] + ["ln_out"],
         f"Each sense has its own encoder (a network that turns raw input into one "
         f"{d_model}-number token): {enc_lines}. All encoders share one output "
         f"LayerNorm (a normalization layer) with no learned scale, so every real "
         f"token leaves at the same size (RMS 1.0) - tokens produced any other way "
         f"once reached the wellness scorer 1,500x off scale. That size holds where "
         f"the token LEAVES the encoder"
         + (", and the top-down gate one station later blends the model's own guess "
            "into it without renormalizing, so the token the core reads is close to "
            "RMS 1 but not held there: across H81's 19 captured token pairs the "
            "audio token arrived between 94.6% and 101.3% of the raw token's size."
            if td_on else ". Nothing downstream rescales it in this run."),
         edges=([[f"sense_{m}", f"enc_{m}"] for m in mods]
                + [[f"enc_{m}", "ln_out"] for m in mods]))

    step("assembly", "The sequence: {n} senses x {c} model steps".format(n=n_mod, c=ctx),
         ["assembly"] + (["td_gate"] if td_on else []),
         f"The current frame's {n_mod} tokens join the last {ctx - 1} remembered "
         f"frames to form a {seq}-token sequence covering {ctx * step_s:.2f} s of "
         f"body time. Two learned embeddings (lookup "
         f"tables added to each token) tag every token with WHICH sense it is and "
         f"WHICH model step it belongs to - the transformer's only way to tell them "
         f"apart. (Both were accidentally frozen at random values until 2026-08-25.) "
         f"The model step tag is the token's PLACE IN THE WINDOW, so it slides: the "
         f"newest slot always means now. Only the newest {n_mod} tokens carry "
         f"learning back to the encoders - the other {seq - n_mod} were stored with "
         f"their gradient cut off. After a reset the window refills from zeros, so "
         f"for the next {ctx - 1} steps the older slots are empty tokens."
         + (" One station earlier, the top-down gate mixes the model's own "
            "prediction of THIS model step into each sense's token: at most half of it, "
            "because the gate is half a sigmoid, so at least half of every token is "
            "always the measurement. Every lesson is still graded against the raw "
            "encoder token." if td_on else ""),
         edges=gate_chain + [["assembly", "block0_s1"]])

    step("stage1", "Stage 1 - senses look at each other",
         [f"block{i}_s1" for i in range(n_layers)],
         f"Inside each of the {n_layers} transformer blocks, stage 1 runs attention "
         f"({n_heads} heads, averaged in the maps here) ACROSS the {n_mod} senses at "
         f"each model step separately: tactile can consult vision about the same instant. "
         f"The inspector's stage-1 heatmap is this stage's weight matrix - row = the "
         f"sense doing the looking, column = the sense looked at. 'At each model step "
         f"separately' says where the attention may LOOK, not what the tokens carry: "
         f"from block 1 (the 2nd) onward the tokens it lines up have already had other model steps "
         f"mixed into them by block 0's unmasked stage 3. The block count is the "
         f"--core-blocks launch setting, {n_layers} here; a launch that leaves it at 0 "
         f"builds the 2-block default.",
         edges=([["assembly", "block0_s1"]]
                + [[f"block{i}_s1", f"block{i}_s2"] for i in range(n_layers)]
                + [[f"block{i - 1}_ffn", f"block{i}_s1"] for i in range(1, n_layers)]))

    step("stage2", "Stage 2 - each sense looks back through time",
         [f"block{i}_s2" for i in range(n_layers)],
         f"Stage 2 runs attention along the {ctx} model steps within each sense "
         f"separately, under a causal mask (a hard rule that no token may attend to "
         f"the future) - the only mask in the block. This is where memory lives: a "
         f"reach or a roll spans dozens "
         f"of these {step_s * 1000:.0f} ms model steps. The mask says where this stage "
         f"may LOOK: from block 1 (the 2nd) onward the tokens it looks at have already had "
         f"later model steps mixed into them by block 0's unmasked stage 3.",
         edges=([[f"block{i}_s1", f"block{i}_s2"] for i in range(n_layers)]
                + [[f"block{i}_s2", f"block{i}_s3"] for i in range(n_layers)]))

    step("stage3", "Stage 3 - the bridge across everything",
         [f"block{i}_s3" for i in range(n_layers)],
         f"Stage 3 lets all {seq} tokens attend to all others - any sense at any "
         f"model step in the {ctx * step_s:.2f} s window, earlier or later. This "
         f"stage carries no mask at all. It is safe because the window ENDS at now: "
         f"every output reads only the newest model step's {n_mod} tokens, so there is "
         f"no later model step inside the window to leak. "
         f"The snapshot stores the current frame's rows plus two "
         f"summaries (sense-pair mean and attention-by-time-distance); the full "
         f"{seq}x{seq} matrix is deferred (plan v2).",
         edges=([[f"block{i}_s2", f"block{i}_s3"] for i in range(n_layers)]
                + [[f"block{i}_s3", f"block{i}_ffn"] for i in range(n_layers)]))

    n_pred = len(receipt.get("prediction_head_modalities") or [])
    step("heads", "Prediction heads and the delayed lesson", ["heads"],
         (f"The prediction heads (small attention + MLP readouts, one per sense "
          f"except motor - {n_pred} of them for this run's {n_mod} senses, because "
          f"the command is an INPUT, not something to predict) output the NEXT "
          f"model step's tokens"
          + (" - in residual mode, they output the CHANGE and the current token is "
             "added, so 'nothing happens' is the zero output" if residual else "")
          + ". The delayed next-frame lesson is the one whose error the dashboard "
            "reports as causal_pe: the PREVIOUS frame "
            "is re-encoded live with the command that was actually executed, run "
            "through the whole stack, and graded against THIS frame's raw tokens "
            "(S(t-1) + A(t-1) -> S(t)). It is one of the ten lessons graded every "
            "step, and how big a share of a given step's teaching signal it "
            "actually carried is printed in the inspector's loss panel - do not "
            "assume it is the largest. A head is graded only on the steps where "
            "its own sense arrived REAL"
          + (": a sense fed by a placeholder generator - intermind, which nothing in "
             "this world produces - runs its head every step and teaches on none of "
             "them, so fewer heads are graded than are built."
             if "intermind" in mods else
             ", and in this run no head's sense is fed by a placeholder generator.")),
         edges=[["final_norm", "heads"]])

    step("recon", "Reconstruction decoders - the guard", ["recon"],
         "Separate small decoders re-produce the CURRENT frame from the transformer "
         "output. They certify the token still contains the present without letting "
         "the copy-task compete with prediction (their weight follows a schedule, "
         "with a floor at 2% of full weight). Their error is recon_dec; the prediction heads never see it.",
         edges=[["final_norm", "recon"]])

    step("motor", "The motor controller and the executed command", ["motor", "gaze"],
         f"The motor controller (an MLP reading the pooled transformer output, the "
         f"{n_handles}-handle body state, and the slope of wellness with respect to "
         f"the body's MEASURED handle state - its 9 position-and-strength numbers per "
         f"handle, not the command the controller is about to give) proposes "
         f"{per_handle} numbers per handle: position, rotation, radius, "
         f"strength. Handle {gaze_idx} of the {n_handles} is the eyes - gaze is a "
         f"limb here, not a special "
         f"pathway. One command is HELD for {hold} model steps (a limb needs ~250 ms "
         f"to obey), and what the ledger records - and the model learns from - is "
         f"the command actually in force, never the raw proposal. What changes it on "
         f"the way out: the growth phase's force limit, and a clamp holding one "
         f"command's move to the phase's share of 40 cm"
         + (f"; a smoothing blend of {round(100 * smooth)}% of the new command with "
            f"{round(100 * (1 - smooth))}% of the previous one" if smooth > 0 else "")
         + ("; and in a rest window of the babble schedule a postural hold that "
            "replaces the command's position and rotation with the body's own "
            "achieved pose, leaving only its strength"
            if (args.get("babble_bouts") == "real"
                and args.get("rest_control") == "scheduled") else "")
         + ". Exploration noise is added only on steps the planner did not author"
         + (", and with --intention-every dividing --motor-hold in this run every "
            "command the body executed was a planner command: noise reached the body on "
            "none of them, and the controller's own proposal only seeded the plan."
            if _every and hold % _every == 0 else
            " - here those are the command boundaries the planner did not land on, "
            "and they are the only commands that reach the body carrying noise."),
         edges=[["final_norm", "motor"], ["intention", "motor"],
                ["motor", "gaze"], ["motor", "world"]]
               # the motor sense's achieved handle state, read without the encoder (the input check of Oct 15, 2026)
               + ([["sense_motor", "motor"]] if "motor" in mods else []))

    step("wellness", "Wellness: measured scores, predictions and goal ranking", ["wellness"],
         "Fixed formulas calculate six scores per body: encoded prediction accuracy, skin comfort, "
         "remaining capacity, body-map co-movement, handle-position boundary score and self-contact increase. "
         "Prediction accuracy compares predicted and measured lists of learned sensory numbers. It squares "
         "their differences so opposite errors cannot cancel, then averages so list length does not inflate the result. "
         "A score of 1 means matching compared numbers, including during motion; it does not mean a still crib. "
         "A separate learned network estimates the first five scores. Its training targets are the fixed scores. "
         "The motor controller receives the derivative of their sum with respect to measured handle coordinates; "
         "only the position formula contributes a nonzero derivative. Goal ranking is a third calculation, "
         "combining enabled channels to choose a remembered sensory target. Click the wellness box for each formula.",
         edges=[["ln_out", "wellness"], ["wellness", "motor"], ["wellness", "intention"]]
               # skin comfort and self-contact read the tactile sense, the handle-position and capability scores the
               # motor sense, both without the encoder (the input check of Oct 15, 2026)
               + [[f"sense_{m}", "wellness"] for m in ("tactile", "motor") if m in mods])

    step("curiosity", "Curiosity: a configured preference over prediction error", ["curiosity"],
         "The drive is zero outside the configured error range, rises in a straight line from zero at its lower "
         "edge to one at its midpoint, then falls to zero at its upper edge. This formula does not measure whether "
         "the observations can be learned. "
         + ("It is recorded without changing remembered-goal value because curiosity-goal is off. "
            if args.get("curiosity_goal", "off") != "on" else
            "Normalized ranking adds curiosity as its own channel. " if args.get("wellness_rank") == "normalized" else
            "Raw ranking adds drive times its configured weight. " if args.get("curiosity_mode") == "additive" else
            "Raw ranking multiplies the stored value by one plus the drive. ")
         + "The fixed wellness formulas do not use curiosity. It scales the wellness lesson only in direct or forward mode. "
         + f"This run uses {args.get('wellness_mode', 'baseline')} mode.",
         edges=[["curiosity", "intention"]])

    # THE PLANNER IN ONE SLIDE (rewritten 2026-10-02 to state input, computation and output; the
    # whole account is the intention box's panel, schema.DOC node.intention). The fixed numbers - the
    # ring's 200 entries, the 20 updates, the step size and the strength weight - are lem_model.py's
    # own (tests/test_build_explainer.py holds them to the model); the rest is the run's launch flags.
    hand = args.get("hand_target_mode") == "wrist_relative_v1"
    normalized = args.get("wellness_rank") == "normalized"
    smooth_new = round(100 * smooth)
    step("intention", "Intention - planning a command toward a goal-ring entry",
         ["intention"],
         f"On 1 model step in {_every or 5}, once the goal ring holds 20 entries, the planner "
         f"computes one command per body: {n_handles} body handles x {per_handle} numbers (target "
         f"position, orientation, radius, strength)"
         + (", plus a target angle for every finger joint (--hand-target-mode wrist_relative_v1)"
            if hand else "")
         + ". The goal ring keeps the last 200 model steps' tokens and wellness values per body; "
         + ("each entry is ranked by its wellness channels, each measured against its running mean "
            "in units of its running spread and weighted by priority (--wellness-rank normalized)"
            if normalized else
            "each entry is ranked by its stored value, the wellness sum"
            + (" scaled or raised by the curiosity drive (--curiosity-goal on)"
               if (args.get("curiosity_goal") or "off") == "on" else ""))
         + ", and the top entry's tokens are the goal. Starting from the motor controller's "
         "command, the planner runs 20 Adam updates (Adam: a gradient optimizer; step size one "
         "tenth) on the command's numbers. Each update puts the command through the execution "
         "filter (reach and strength limits"
         + (f", a blend of {smooth_new} parts the command with {100 - smooth_new} parts the previous one"
            if smooth > 0 else "")
         + "), runs the frozen forward model once to predict the next model step's tokens, scores the "
         "prediction - for each live sense the mean squared difference from the goal tokens, added "
         "over the senses, plus one hundredth of the summed squared strengths - and changes the "
         "command against the score's derivative. No weight changes. The lowest-scoring of the 21 "
         "commands is kept"
         + ("; under --hand-goals memory a second search then sets the finger targets toward the "
            "goal entry's recorded hand pose" if args.get("hand_goals") == "memory" else "")
         + ". If a second forward run predicts a token below half or above twice the size of this "
           "model step's, the plan is dropped and the body gets the motor controller's command. "
         + ("Every plan is one command long under --hand-target-mode wrist_relative_v1. " if hand else
            "Longer plans are licensed depth by depth: depth k + 1 is open while the imagination "
            "chain beats predicting no change at depth k more than half the time; the plan_license "
            "record in the run tab shows how many are open. "
            if args.get("plan_horizon") == "earned" else
            "Every plan is one command long (--plan-horizon fixed). ")
         + "The intention box opens the whole computation with its diagram.",
         edges=[["wellness", "intention"], ["intention", "motor"],
                # the tokens the goal ring stores and the search starts from: after the top-down gate when it is on
                ["td_gate" if td_on else "ln_out", "intention"],
                # the controller's command each search starts from (the independent review of Oct 15, 2026, S1)
                ["motor", "intention"]]
               # the goal ring's stored pose under either memory goal flag
               + ([["sense_motor", "intention"]] if "motor" in mods and "memory" in (args.get("hand_goals"),
                                                                                    args.get("physical_goals")) else []))

    imag_on = (args.get("imagination") or "config") != "off"
    imag_d = args.get("imagination_steps")
    imag_back = (f"{imag_d} steps back" if isinstance(imag_d, int) and imag_d > 0
                 else "as far back as the chain keeps (the launch did not record "
                      "--imagination-steps, so the depth is the one in the config "
                      "file, which this page does not read)")
    # the lesson weight gets the same treatment as the depth: state it when the run
    # recorded it, and say where it came from instead of printing a guessed constant
    imag_w = args.get("imagination_weight")
    imag_wtxt = (f" The lesson's weight in the total is {100 * float(imag_w):g}% of full weight."
                 if isinstance(imag_w, (int, float)) else
                 " Its weight in the total is the one in the config file: this "
                 "launch did not pass --imagination-weight, and the page does not "
                 "read that file.")
    step("imagination", "Imagination - training multi-step accuracy", ["imagination"],
         (f"Inside the live step the model takes the model step {imag_back}, predicts "
          f"the next one, feeds that prediction back in as if it were a real token, "
          f"and predicts again - once per depth - grading every depth against the "
          f"RAW tokens that really arrived.{imag_wtxt} Only senses measured at that "
          f"model step are "
          f"graded and fed back; one that was not measured is put back as it was "
          f"observed, so its noise cannot compound. It reads no stored ring of old "
          f"model steps - that is the replay memory box. No command leaves this chain, so "
          f"it puts no force on the body itself. "
          + (f"In this run it decides nothing about behaviour either: under "
             f"--hand-target-mode {args.get('hand_target_mode')} the planner's search looks one "
             f"model step ahead (lem_model.py _plan_chain sets max_steps = 1), so the depth "
             f"this chain earns is computed and not used."
             if (args.get("hand_target_mode") or "world") != "world" else
             f"It does decide one thing about "
             f"behaviour: how often each depth beats assuming nothing changed sets how "
             f"many model steps ahead the planner may aim, and the planner's first step is "
             f"what the body executes.")
          if imag_on else
          "Switched off for this run (--imagination off): no multi-step lesson was "
          "trained, so nothing here earned the planner any extra depth. The H23-H28 "
          "runs were read as if this chain were training when it was off."),
         edges=([["ln_out", "imagination"], ["motor", "imagination"]]
                + ([["imagination", "intention"]]
                   if (args.get("hand_target_mode") or "world") == "world" else [])
                if imag_on else []))

    ring_need = max(4 * replay_batch, 8 * ctx)
    step("replay", "Replay memory - re-study", ["replay"],
         f"Every {replay_every} live steps the model takes a SECOND, complete pass - "
         f"its own forward, its own loss, its own backward and its own optimizer "
         f"step - over UP TO "
         f"{replay_batch} remembered windows - fewer whenever a drawn window would "
         f"cross an episode reset, and the pass is skipped outright below "
         f"{max(2, replay_batch // 4)} of them. Its loss is NOT the live loss: it is "
         f"the prediction heads graded against the real next model step plus the motor "
         f"controller graded against the real next achieved pose, and nothing else - "
         f"no anti-collapse term, no imagination, no delayed next-model-step lesson, no "
         f"reconstruction, no wellness, no tenure. It re-teaches the forward model, "
         f"the prediction heads, the motor controller and the persistent memory's "
         f"bias readout. Five of the six encoders are excluded, because their tokens were "
         f"stored detached. The MOTOR encoder is NOT excluded: each window's motor "
         f"token is rebuilt from the achieved pose plus the command actually executed, "
         f"so that encoder is trained from memory too. "
         + ("This run was launched --eval-only, which never builds the ring: this "
            "pathway ran 0 times here and moved no weight, and its stats carry no "
            "replay loss at all - any replay loss on the Run tab belongs to the "
            "run the page header names for the charts. "
            if args.get("eval_only") else
            f"With --fill-ring-first it waits until {ring_need:,} stored model steps "
            f"carry the command that left them, about {ring_need * step_s:.0f} s of "
            f"body time. ")
         + "This pathway is invisible in the per-step loss panel by construction; "
           "its own loss is charted in the run tab.")

    if mem_on:
        from lem.model.persistent_memory import MemoryHistory
        mh = MemoryHistory()      # lem_model.py builds it with exactly these defaults
        step("memory", "Persistent memory - beyond the window", ["memory"],
             f"The transformer reads only the last {ctx} model steps (about "
             f"{ctx * step_s:.0f} s of body time). The persistent memory keeps "
             f"striking events beyond that: it retains each half second's biggest "
             f"change, up to {mh.capacity} of them, which is "
             f"{mh.capacity * mh.interval_s:.0f} s of body time. It summarizes that "
             f"whole window in sequence with a recurrent network (a GRU) and weights "
             f"the window by how closely each running summary matches the present "
             f"model step. What it recalls is added to the newest model step's {n_mod} "
             f"tokens AFTER the transformer's last normalization layer has run, so "
             f"it never passes through an attention stage and the prediction heads, "
             f"the "
             f"motor controller, the voice head and the wellness scorer read it and "
             f"the transformer itself does not. The readout is created at zero, so "
             f"installing the memory on an existing checkpoint changes nothing "
             f"until it has trained; once trained it is never zero.",
             edges=[["memory", "final_norm"]])

    if self_mode != "off":
        step("self_state", f"Self perception ({self_mode})", ["self_state"],
             "The model learns a sense of its own body's condition: for each "
             "observed skin site and each actuator group, how well it works right "
             "now, and what a run of commands did to it. The commands it rolls through "
             "are ones the body ALREADY EXECUTED, replayed from a store of the last "
             "128 and graded against measurements that have already arrived - one "
             f"command is one model step, {step_s:.2f} s of body time. Nothing here "
             "predicts past now unless --self-choice is on"
             + (" - it is off in this run. " if (args.get("self_choice") or "off") == "off"
                else ". ")
             + "It predicts CHANGE from the observed present, so predicting 'nothing "
             "changes' is the zero output. "
             + ("In 'learn' mode its belief reaches no other part of the model - no "
                "perception bias, no goal, no command. " if self_mode == "learn"
                else "In 'perceive' mode its belief also feeds the forward model. ")
             + ("This page's run was launched --eval-only: these lessons were "
                "computed and reported and no weight moved."
                if args.get("eval_only") else
                "These lessons update this module's own weights every step while the "
                "run is training."))

    if voice_on or caregiver:
        step("voice", "The voice head and the caregiver",
             (["voice"] if voice_on else []) + (["caregiver"] if caregiver else []),
             (f"A voice head reads the transformer's output and emits 80 "
              f"numbers each step, each held between 0 and 1 by a sigmoid. 80 "
              f"is the width of the action slot the shared-memory buffer "
              f"reserves, not a vocal tract's control count - the repo's only "
              f"vocal-tract parameter set is 56 numbers (11 parameters, a "
              f"wrapped synth clock, 44 tract diameters), so 24 of the 80 have "
              f"no named meaning. No lesson reaches it: no "
              f"target is ever given and its loss weight is zero, so its "
              f"{int((receipt.get('voice_head') or {}).get('params') or 0):,} "
              f"numbers still hold the values they were first given, and the "
              f"controls sit where an untaught readout sits - close to the middle of their 0-to-1 range. "
              f"Voice is not implemented in the MIMo world yet - the controls "
              f"make no sound and the model cannot hear itself; it gets wired "
              f"once BioFold, the physical vocal-tract simulation, is done. "
              if voice_on else "")
             + (f"A caregiver lives in the world. The two launch flags time the "
                f"TOY, not the adult: a toy stays in reach "
                f"{float(args.get('caregiver_present_seconds') or 0):g} s, then "
                f"{float(args.get('caregiver_absent_seconds') or 0):g} s pass before "
                f"the next one arrives. The adult's own comings and goings are "
                f"drawn at random - here 180 to 300 s, then gone 67 to 97 s. "
                f"While present they bring and remove toys, speak short lines "
                f"rendered ahead of time by a text-to-speech voice, go quiet for "
                f"a stretch, and answer what the baby DOES - a look held 0.4 s "
                f"on their face or on the toy, any touch of the toy, or the toy "
                f"shifting 1.5 cm in half a second while the baby watches it. "
                f"While the adult is away nothing the baby does is answered. The "
                f"world decides these comings and goings, not the model - never "
                f"read an absence as the model's intention."
                if caregiver else ""),
             edges=[["final_norm", "voice"]] if voice_on else [])

    if hself_on:
        from . import schema as _SCH
        _slots = int(args.get("hself_slots") or 64)
        _climb = _SCH.hself_first_evidence_windows(_slots)
        _when = ("nothing can ever count on a table this big" if _climb is None else
                 f"a pattern counts only after beating 8 fake wellness signals for {2 + _climb} windows "
                 f"- {(2 + _climb) * 100_000:,} model steps for Level 3")
        step("hself", "Hierarchical self - watching, not acting", ["hself"],
             f"Wellness scores the present model step. This part measures what is "
             f"good to HAVE AROUND - things whose presence goes with wellness over minutes (Level 2 of "
             f"the self) and over hours (Level 3). It stores kinds of model steps in {_slots:,} slots; a "
             f"slot is one stored pattern of what all six senses reported at one model step. Each model "
             f"step it computes how alike that model step and each stored pattern are (a cosine per "
             f"sense), and it records whether a pattern's "
             f"coming and going matches the rise and fall of wellness. A match can be luck, so "
             f"{_when}. Until then its count shows -1, 'cannot be measured yet'. It is 'dark': it "
             f"only records, and nothing it computes changes what the model does. Click the box "
             f"for the five steps and the warnings.",
             edges=[["wellness", "hself"], ["ln_out", "hself"]])

    # ---------------- the gain-of-function build (2026-09-22): one slide per feature this run
    # has on. Each names only boxes buildDiagram draws for the same receipt and flags, and
    # animates only lines it draws (tests/test_build_explainer.py holds both).
    if (args.get("entity_self") or "off") != "off":
        step("entity_self", "Named things the self-model watches", ["entity_self"],
             "The same job as the hierarchical self - learning what is good to HAVE AROUND - "
             "over things that can be named: the caregiver, each toy, the mobile, the squeaker, "
             "the critters and the baby's own hand in view. Each has its own presence signal "
             "read from the world's report (0 = not there, 1 = there and squarely in view or in "
             "hand), and each body's presence is correlated with that body's wellness over "
             "minutes and over hours, against 8 sign-flipped controls. A thing counts only "
             "after beating its controls for enough windows. Then the reader may add, to a "
             "remembered model step's value, its weight x the sum of (proven link x presence), so "
             "the planner aims at model steps that held proven things - exactly 0 until the "
             "reader's own gate opens.",
             edges=[["wellness", "entity_self"], ["entity_self", "intention"]])
    if "vestibular" in mods:
        step("vestibular", "Vestibular - balance, neck, support, effort",
             ["sense_vestibular", "enc_vestibular"],
             "Behind --vestibular-sense on, the sixth input slot - where the intermind "
             "placeholder's noise used to go - carries the body's own load and balance: which "
             "way gravity pulls in the head and how fast the head turns, the neck's angles, "
             "speeds and efforts, which way is up for the trunk, what share of the body's "
             "support each part carries, mechanical effort, joints at their limits and "
             "fatigue - 29 numbers, each scaled so the crib body's 99th percentile reads about "
             "1. It is graded like every other real sense, so the model now has to predict "
             "its own posture's consequences.",
             edges=[["world", "sense_vestibular"], ["sense_vestibular", "enc_vestibular"],
                    ["enc_vestibular", "ln_out"]])
    if receipt.get("efference_heads"):
        # the gaze copy (lem/model/efference.py GazeCopy): the heads' stamp records it whenever the
        # model has vision, and viewer.js then draws its line into the prediction heads
        eff = receipt.get("efference_heads")
        gaze_copy = isinstance(eff, dict) and bool(eff.get("gaze_copy_into_vision"))
        step("efference", "Efference heads - the command, given directly",
             ["efference"] + (["heads"] if gaze_copy else []),
             "The executed command reaches the core only as 9 of 19 columns per handle, pooled "
             "into one motor token, and over all of H84 knowing it improved the next-model-step "
             "prediction by at most 0.08% of the way from no use to exact. These two heads are handed "
             "the command DIRECTLY, beside the core's state, and predict what it will do in "
             "units a person can picture: every handle's move in cm and turn in degrees over "
             "the next 50 ms and over the command's 250 ms, and the eyes' turn in degrees at 50, "
             "100 and 150 ms. Their next-model-step error trains the core too. Each is graded "
             "beside doing nothing and beside another body's command size-matched to this "
             "body (its asked moves and turns, measured from its own pose, applied to this "
             "body's pose; for one segment also beside its absolute command)."
             + (" A third path, the gaze copy, adds the eye command's expected effect to the "
                "prediction heads' vision token, so the predicted next view - in its lessons "
                "and in the planner's imagined steps - depends on where the eye command "
                "points." if gaze_copy else ""),
             edges=[["motor", "efference"], ["final_norm", "efference"]]
                   + ([["efference", "heads"]] if gaze_copy else []))
    # WHERE THE HEADS REACH THE PLANNER (schema.efference_routes: the rule viewer.js draws the two
    # lines out of the efference box by). Only when this run has at least one of the routes.
    eff_routes = efference_routes(receipt, args)
    if eff_routes["goal_channels"] or eff_routes["action_search"]:
        chans = eff_routes["goal_channels"]
        eff_nodes, eff_edges, parts = ["efference"], [], []
        if chans:
            eff_nodes += ["wellness", "intention"]
            eff_edges += [["efference", "wellness"], ["wellness", "intention"]]
            parts.append(
                "The planner's goal is the remembered model step it aims at, and this run chooses it by "
                "ranking the wellness channels stored with every model step (--wellness-rank "
                "normalized). "
                + ("Two of those channels read the heads." if len(chans) == 2 else
                   "One of those channels reads the heads.")
                + (" Controllability sets the body head's prediction for the whole command, made when "
                   "the command was sent, against what the body did; it counts only while its evidence "
                   "gate is open." if "controllability" in chans else "")
                + (" Learning progress takes in each head's error against doing nothing, beside "
                   "every graded sense." if "learning_progress" in chans else ""))
        if eff_routes["action_search"]:
            if "intention" not in eff_nodes:
                eff_nodes.append("intention")
            eff_edges.append(["efference", "intention"])
            parts.append(
                "Under --physical-goals memory the planner " + ("also " if chans else "")
                + "calls the body head while it searches for a command: it scores each candidate by how "
                  "far the body head's predicted pose lands from the pose stored with the goal.")
        step("efference_planner", "What the efference heads change in the planner", eff_nodes,
             " ".join(parts), edges=eff_edges)
    if receipt.get("contact_head"):
        step("contact_head", "The contact head - foreseeing touch",
             ["contact_head"],
             "For each of the 53 skin patches: will anything touch it in the next model step, and "
             "is it an own part or an outside thing? A touch that begins counts as FORESEEN "
             "only when the head's probability beat the patch's base rate of starting that "
             "touch by a measured margin. Foreseen self-touch pays twice what a foreseen "
             "outside touch pays, a repeated pair pays less each time, and nothing pays until "
             "foreseeing with the body's own command beats foreseeing with another body's command. "
             "Touch merely happening never pays.",
             edges=[["motor", "contact_head"], ["final_norm", "contact_head"]])
    if receipt.get("motor_synergy"):
        step("synergy", "The motor synergy - a compact intention",
             ["synergy"],
             "A small network turns a few numbers of intention into a coordinated change of "
             "every body handle's target, trained on the posture changes the body really "
             "made. The planner adjusts those few numbers instead of 234 independent ones, "
             "and babble can draw them too, because the body follows a posture it has really "
             "reached far better than 25 independent turns.",
             edges=[["synergy", "intention"]])
    channels_on = [k for k in ("controllability", "learning_progress") if args.get(k) == "on"]
    if channels_on or args.get("wellness_rank") == "normalized":
        step("wellness_channels", "Wellness channels: ranking remembered sensory states",
             ["wellness", "intention"],
             "Additional channels include predicted-versus-measured command effects, recent prediction-error trends, "
             "and correctly predicted new contacts. Normalized ranking stores each channel minus that body's recent "
             "average, then subtracts a running centre, divides by a running spread or minimum scale, multiplies by "
             "its configured weight and adds the results. It ranks changes relative to recent history, not absolute "
             "wellness. Curiosity is a separate channel when enabled. Controllability and contact control have "
             "their own evidence gates; this rule does not apply to every channel. A memory above the harm limit "
             "is excluded while a safer one is available. Click wellness for the formulas and current settings.",
             edges=[["wellness", "intention"]])
    if args.get("causal_chain") == "on":
        step("causal_chain", "The causal chain, recorded", ["intention", "motor", "world"],
             "Every planning event and every command sent to the body is recorded as five "
             "links: the intention (the remembered model step aimed at), the prediction made "
             "BEFORE the command was sent, the predicted wellness of the chosen command against "
             "four alternatives from the same state, whether the command reached the body, and "
             "the result one model step later and at the command's end - graded against the "
             "prediction and against 'nothing changes'. The inspector's causal chain panel "
             "and the run tab's causal-chain charts read it.",
             edges=[["intention", "motor"], ["motor", "world"]])

    spp = args.get("steps_per_phase")
    step("curriculum", "The developmental curriculum", ["curriculum"],
         "Six phases - neonate, early infant, late infant, toddler, child, mature "
         "- raise how hard the body may push (2% of full force to 100%) and how far "
         "one command may move a handle (2 cm to 40 cm), shorten the wait before a "
         "command acts (30 ms to 0) - no sense is delayed at any phase: the bridge "
         "reads live MuJoCo sensors and the world refuses a sensory delay - and set "
         "the simulated age the world workers regrow the body at (1 month to 24 "
         "months). Babble noise is the one thing they do NOT change: the table "
         "falls from a 1 cm random offset to none and a floor lifts every phase to a random offset "
         "of 5 cm of position and 29 degrees of rotation on a command, one "
         "standard deviation - so the trainer receives the same babble in all "
         "six. "
         + (f"This run gives each phase {int(spp):,} model steps. " if spp else "")
         + "A phase change that changes the body rebuilds the worlds, and the "
         "context window, the pending lessons and the goal ring are dropped, so no "
         "single lesson pairs one body's cause with another body's effect - the "
         "model steps already in the replay ring are not dropped and keep being "
         "re-studied.",
         edges=[["curriculum", "motor"], ["curriculum", "world"]])

    step("losses", "What trains what", ["heads", "recon", "motor", "wellness"],
         "Ten lessons are graded every step: the delayed next-frame lesson (its "
         "error is causal_pe), the reconstruction decoders' guard, the motor "
         "lesson (grade the PREVIOUS command against the pose it produced), the "
         "wellness-alignment lesson (the learned copy of the wellness scorer), "
         "vicreg (variance-invariance-covariance regularization - an "
         "anti-collapse term keeping tokens spread out), the latent regulariser, "
         "imagination, the anticipation-gaze lesson, the persistent memory's "
         "next-needs forecast, and the self-perception lesson, which is the "
         "largest single term in this run's total and the reason that total is "
         "negative. At a command boundary two more join while training: the "
         "action-tenure pose hinge and the command-to-outcome matching term "
         "(tenure and tenure_tok - does the executed command explain the outcome "
         "it claims, the gain-of-function signal rank1 tracks). A prediction "
         "term per sense also exists but is weighted zero whenever "
         "--next-frame-loss is on. Every term's live value is in the "
         "inspector's loss panel."
         + (" This run adds the build's own lessons: "
            + "; ".join(t for t in (
                "the efference heads' body and gaze lessons (each at the weight "
                "--efference-weight sets)" if receipt.get("efference_heads") else "",
                "the contact head's lesson (at the weight --contact-head-weight sets)"
                if receipt.get("contact_head") else "",
                "the motor synergy's hindsight lesson" if receipt.get("motor_synergy") else "")
                if t) + "."
            if (receipt.get("efference_heads") or receipt.get("contact_head")
                or receipt.get("motor_synergy")) else ""),
         # wellness is NOT on the core's output line: the learned scorer reads the
         # RAW encoder tokens off the shared LayerNorm, so its line starts there
         edges=([["final_norm", t] for t in ("heads", "recon", "motor")]
                + [["ln_out", "wellness"]]))

    return s
