# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Consequence localization: WHERE does wellness-directed choice break first?

The 09-09 follow-up review's step 2, verbatim goal: from identical saved
states, compare candidate actions at 50 ms, the held 250 ms order, and 1 s;
record predicted and realized touch/capacity/posture, action differences after
execution constraints, and the gradients of the competing planner objectives -
then say whether failure first appears in SENSORY INFORMATION (consequences
don't differ), PREDICTION (imagined next state ranks candidates wrongly),
VALUATION (the judge mis-scores real states), or ACTION OPTIMIZATION (the
constrained/smoothed actions barely differ, or the chosen one ranks low).

Reuses the planner's OWN canonical machinery - predictive_state.transition,
the differentiable execution transform, the shared smoother - so the probe
cannot quietly diverge from what planning computes. Uncertainty is reported
BY WORLD (independent env seeds), never by treating adjacent frames as
independent (review requirement).

python -m mimo_bridge.tools.consequence_localization --ckpt <ckpt_dir>
    [--worlds 3] [--events 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
SAVED = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "Saved", "mimo_bridge")

from mimo_bridge.tools.causal_test import _build_model, _limits_for_run  # noqa: E402
from mimo_bridge.tools.sensor_action_contract_check import (  # noqa: E402
    resolved_env_kwargs, hold_current)
from mimo_bridge.tools.first_action_benefit import (  # noqa: E402
    physics_state, restore_physics, state_wellness, encode_frame,
    matched_control, free_ram_gb)
from mimo_bridge.lem_model import LemLite  # noqa: E402
from mimo_bridge.tools.probe_clock import ProbeClock

HORIZON_SECONDS = (.05, .25, 1.)


def _decision_state(live, raw=False):
    """Frame t paired with the context BEFORE frame t was appended."""
    from lem.model.predictive_state import PredictiveState
    saved = live._last_plan_inputs.get('decision') or {}
    context = saved.get('context', live.last_context)
    if context is None:
        raise ValueError('decision has no recorded pre-shift context')
    tokens = live.last_tokens if raw else live._last_plan_inputs['cur']
    return PredictiveState(
        tokens={m: v.reshape(v.shape[0], -1).detach().clone() for m, v in tokens.items()},
        context=context.detach().clone(),
        achieved=saved.get('achieved', live.last_achieved).detach().clone(),
        source_mask=dict(saved.get('source_mask', live._last_source_mask) or {}),
        memory_bias=None if saved.get('memory_bias') is None else saved['memory_bias'].detach().clone())


def _predict_candidate(live, exec_cmd_cm: torch.Tensor, raw=False, capture_heads=False):
    """Predicted next-state tokens for one EXECUTABLE candidate, through the
    planner's canonical route (transition + normalize_executed_command)."""
    from lem.model.predictive_state import PredictiveState, transition
    from lem.model.encoder_registry import normalize_executed_command
    state = _decision_state(live, raw=raw)
    action = normalize_executed_command(exec_cmd_cm)
    captured = {}
    hook = None
    if capture_heads:
        def capture(module, args, result):
            captured.update({m: v.detach().clone() for m, v in result.items()})
        hook = live.prediction_heads.register_forward_hook(capture)
    try:
        with torch.no_grad():
            pred = transition(live.forward_model, live.encoder_registry,
                              live.prediction_heads, state, action,
                              advance_context=False).tokens
    finally:
        if hook is not None:
            hook.remove()
    # Training grades the head outputs before transition's extra output norm.
    # Capture that exact tensor to distinguish prediction from planning-path changes.
    if capture_heads:
        captured['motor'] = pred['motor']
        return pred, captured
    return pred


def _judge_bodily(live, tokens: dict) -> float:
    """Judge B's bodily columns (touch, capacity, posture) on a token dict."""
    zm = tokens.get("motor")
    if zm is None:
        zm = torch.zeros(1, live.d_model)
    sens = {m: (v if v.dim() == 2 else v.squeeze(1)) for m, v in tokens.items()
            if m != "motor"}
    with torch.no_grad():
        wj = live.wellness.score_predicted_wellness(sens, zm)
    return float(wj[:, [1, 2, 4]].mean())


def _objective_grads(live, dev) -> dict:
    """Gradient norms of the two competing planner objectives at the STORED
    decision (goal-prediction term vs prospective-wellness term), through the
    canonical differentiable route - the planner's own arithmetic."""
    from lem.model.predictive_state import PredictiveState, transition
    from lem.model.encoder_registry import normalize_executed_command
    from lem.model.intention_generator import (
        _dict_mse, prospective_wellness_loss, _freeze_params, _restore_params)
    pi = live._last_plan_inputs
    saved = pi.get('decision') or {}
    plan0 = pi["plan0"].detach().clone().requires_grad_(True)
    state = _decision_state(live)
    frozen = [live.forward_model, live.encoder_registry, live.prediction_heads,
              live.wellness]
    prior = _freeze_params(frozen)
    try:
        with torch.enable_grad():
            action = LemLite.planned_execution(
                plan0, saved.get('dev', dev), handle_state=state.achieved,
                previous=saved.get('smooth_prev', getattr(live, '_last_smooth_prev', None)),
                smoothing=saved.get('smoothing', live.smooth_actions))
            action = normalize_executed_command(action)
            pred = transition(live.forward_model, live.encoder_registry,
                              live.prediction_heads, state, action,
                              achieved_state=state.achieved,
                              advance_context=False).tokens
            goal = pi["goal"]
            obj = {m: v for m, v in pred.items()
                   if m != "motor" and m in goal
                   and str(getattr(state.source_mask.get(m, 'real'), 'value',
                                   state.source_mask.get(m, 'real'))) == 'real'}
            pred_loss = _dict_mse(obj, {m: goal[m] for m in obj})
            g_pred = torch.autograd.grad(pred_loss, plan0,
                                         retain_graph=True)[0]
            wfn = live.intention_gen._wellness_score_fn
            g_well = None
            if wfn is not None:
                wl = prospective_wellness_loss(pred, wfn, 1.0)
                if wl.requires_grad:
                    g_well = torch.autograd.grad(wl, plan0,
                                                 allow_unused=True)[0]
    finally:
        _restore_params(prior)
    return {"goal_term_grad_norm": float(g_pred.norm()),
            "wellness_term_grad_norm_at_w1":
                (float(g_well.norm()) if g_well is not None else None),
            "meaning": "per-unit-weight gradient sizes of the two objectives "
                       "on the final raw action, using the planner's real-sense mask. "
                       "This norm ratio is not an estimate of Adam's action change."}


def _rank_agreement(a: list, b: list) -> float:
    """Kendall-style pairwise agreement in [-1, 1] between two score lists."""
    n = len(a)
    num = den = 0
    for i in range(n):
        for j in range(i + 1, n):
            sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if sa and sb:
                num += int(sa == sb) - int(sa != sb)
                den += 1
    return num / den if den else float("nan")


def run_world(ckpt: str, world: int, events: int, a, live, meter, limits) -> dict:
    from mimo_bridge.datasource import SingleEnvSource
    reps = int(a.get("action_repeat") or 5)
    hold = int(a.get("motor_hold") or 5)
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    ckpt_phase = None
    try:
        cs = json.load(open(os.path.join(ckpt, "meta.json"))).get("curriculum_state") or {}
        ckpt_phase = cs.get("phase")
        if ckpt_phase == "fixed":
            ckpt_phase = None
    except OSError:
        pass
    env_kw, env_phase = resolved_env_kwargs(run_dir, phase_override=ckpt_phase)
    env_kw = dict(env_kw, seed=int(env_kw.get("seed") or 0) + 7000 * world)
    src = SingleEnvSource(**env_kw)
    clock = ProbeClock(float(src.env.dt), reps, hold)
    horizon_steps = {clock.steps_for(sec): sec for sec in HORIZON_SECONDS}
    if not np.isclose(clock.model_step_seconds, HORIZON_SECONDS[0]):
        src.close()
        raise ValueError('this one-step comparison requires a 50 ms trained model step')
    dev = dict(limits)
    dev["exploration_noise"] = float(a.get("babble_noise") or 0.0)
    meter_dev = dict(limits, exploration_noise=0.0)
    rng = np.random.default_rng(3100 + world)
    rows = []
    try:
        live.reset_transition_state()
        frames = src.reset()
        held, prev_exec = None, None
        i = 0
        while len(rows) < events and i < 120 + events * 40 * 3:
            with torch.no_grad():
                res = live.step(frames, dev, train=False, step_index=i,
                                executed_action=prev_exec)
            if i % hold == 0 or held is None:
                held = res.motor_for_action
            li = live.last_intention
            pi = getattr(live, "_last_plan_inputs", None)
            if (i >= 120 and i % 40 == 0 and li is not None and pi is not None
                    and len(live.goal_ring) >= 20):
                plan0 = pi.get("plan0")
                if plan0 is None:
                    i += 1
                    continue
                ach_t = live.last_achieved.detach().float()
                ach_cm = (ach_t[:, :, 0:3].cpu().numpy() * 200.0)
                sm_prev = getattr(live, "_last_smooth_prev", None)

                def exec_form(raw_t):
                    with torch.no_grad():
                        return LemLite.smooth_blend(
                            live.execution_transform(raw_t.to(live.device), dev,
                                                     handle_state=ach_t),
                            sm_prev, live.smooth_actions)

                chosen_exec = exec_form(plan0)
                cands = {"chosen": chosen_exec,
                         "replay": chosen_exec,
                         "hold": torch.as_tensor(
                             hold_current(frames[0].bones)[0][None],
                             dtype=torch.float32),
                         "alt1": torch.as_tensor(matched_control(
                             chosen_exec.cpu().numpy(), ach_cm, rng),
                             dtype=torch.float32),
                         "alt2": torch.as_tensor(matched_control(
                             chosen_exec.cpu().numpy(), ach_cm, rng),
                             dtype=torch.float32)}
                # The controls rotate the already executable command. They are
                # physical interventions, not separately optimized raw actions.
                names_d = [k for k in cands if k != "replay"]
                dif = [float((cands[x] - cands[y]).abs().max())
                       for xi, x in enumerate(names_d) for y in names_d[xi + 1:]]
                grads = _objective_grads(live, dev)
                # PREDICTED next-state (50 ms) judged wellness per candidate
                pred_judged, pred_tokens, training_tokens = {}, {}, {}
                for k, c in cands.items():
                    if k == "replay":
                        continue
                    pred_tokens[k] = _predict_candidate(live, c)
                    _, training_tokens[k] = _predict_candidate(live, c, raw=True,
                                                               capture_heads=True)
                    pred_judged[k] = _judge_bodily(live, pred_tokens[k])
                # REALIZED rollouts
                state0 = physics_state(src.env)
                sim_start = float(frames[0].sim_time)
                wb = state_wellness(meter, frames[0])
                realized = {}
                for k in rng.permutation(list(cands.keys())):
                    restore_physics(src.env, state0)
                    bf = frames
                    cmd = cands[k].cpu().numpy().reshape(1, -1, 9)
                    arm_gaze = live._gaze_from_handle(
                        torch.as_tensor(cmd, dtype=torch.float32,
                                        device=live.device))
                    rec, wmin = {}, float("inf")
                    for fr in range(1, max(horizon_steps) + 1):
                        bf = src.step(cmd, arm_gaze, None,
                                      render=(fr in horizon_steps))
                        sw = state_wellness(meter, bf[0])
                        wmin = min(wmin, sw["total"])
                        if fr in horizon_steps:
                            sec = horizon_steps[fr]
                            ms = round(sec * 1000)
                            elapsed = clock.verify_elapsed(sim_start, float(bf[0].sim_time), sec)
                            rec[f"w_{ms}"] = sw
                            rec[f"elapsed_{ms}"] = elapsed
                            if ms == 50:
                                rec["tok_50ms"] = encode_frame(
                                    meter, bf, meter_dev)
                    rec["wmin"] = wmin
                    realized[k] = rec
                restore_physics(src.env, state0)
                # PREDICTION fidelity at 50 ms: does the imagined next state
                # sit closer to the realized one than persistence does?
                pred_fid, train_fid = {}, {}
                for k in pred_tokens:
                    rt = realized[k]["tok_50ms"]
                    mods = [m for m in ("tactile", "proprioception")
                            if m in pred_tokens[k] and m in rt]
                    pmse = float(np.mean([torch.mean(
                        (pred_tokens[k][m][0] - rt[m][0]) ** 2).item()
                        for m in mods]))
                    cur = {m: (v.squeeze(1) if v.dim() == 3 else v)
                           for m, v in live._last_plan_inputs["cur"].items()}
                    pers = float(np.mean([torch.mean(
                        (cur[m][0] - rt[m][0]) ** 2).item() for m in mods]))
                    pred_fid[k] = {"pred_mse": pmse, "persist_mse": pers}
                    train_fid[k] = {
                        'pred_mse': float(np.mean([torch.mean(
                            (training_tokens[k][m][0] - rt[m][0]) ** 2).item() for m in mods])),
                        'persist_mse': float(np.mean([torch.mean(
                            (live.last_tokens[m][0] - rt[m][0]) ** 2).item() for m in mods]))}
                # Candidate EFFECTS without Judge B: can prediction explain how
                # outcomes differ between two actions? Zero effect is the control.
                effects = []
                for xi, x in enumerate(names_d):
                    for y in names_d[xi + 1:]:
                        for m in ('tactile', 'proprioception'):
                            target_delta = (realized[x]['tok_50ms'][m]
                                            - realized[y]['tok_50ms'][m])
                            effects.append({'a': x, 'b': y, 'sense': m,
                                'realized_delta_mse': float(target_delta.square().mean()),
                                'training_delta_error': float(((training_tokens[x][m]
                                    - training_tokens[y][m]) - target_delta).square().mean()),
                                'planning_delta_error': float(((pred_tokens[x][m]
                                    - pred_tokens[y][m]) - target_delta).square().mean())})
                # VALUATION fidelity: the judge on the REALIZED 50 ms tokens
                # vs the grounded formulas on the same frame
                val = {k: {"judge": _judge_bodily(live, realized[k]["tok_50ms"]),
                           "formula": realized[k]["w_50"]["total"] / 3.0}
                       for k in realized if k != "replay"}
                row = {"step": i, "candidate_exec_max_pairwise_diff": dif,
                       "objective_grads": grads,
                       "pred_judged_50ms": pred_judged,
                       "prediction_fidelity_50ms": pred_fid,
                       "training_prediction_fidelity_50ms": train_fid,
                       "action_effects_50ms": effects,
                       "valuation_50ms": val,
                       "w_before": wb}
                for k, rec in realized.items():
                    for sec in HORIZON_SECONDS:
                        ms = round(sec * 1000)
                        row[f"w_{k}_{ms}ms"] = (rec[f"w_{ms}"]["total"]
                                                     - wb["total"])
                        row[f"elapsed_{k}_{ms}ms"] = rec[f"elapsed_{ms}"]
                    row[f"wmin_{k}"] = rec["wmin"]
                rows.append(row)
                print(f"world {world} event {len(rows)}/{events} step {i}",
                      flush=True)
            gz = []
            for r in range(reps):
                frames = src.step(held, res.gaze_out, None, render=(r == reps - 1))
                gz.append(np.stack([np.asarray(f.meta.get("gaze_cmd") or (0.0, 0.0),
                                               dtype=np.float32) for f in frames]))
            prev_exec = LemLite.gaze_to_handle(held, np.mean(np.stack(gz), axis=0))
            i += 1
    finally:
        src.close()
    return {"world": world, "env_seed": env_kw["seed"], "rows": rows,
            'physics_step_seconds': clock.physics_step_seconds,
            'model_step_seconds': clock.model_step_seconds,
            'held_order_seconds': clock.held_order_seconds,
            "env_phase": env_phase}


def summarize(worlds: list) -> dict:
    """Per-world means first, THEN across-world spread (review requirement:
    uncertainty by world, never by adjacent frames)."""
    names = ("chosen", "hold", "alt1", "alt2")
    out = {}
    # sensory: realized spread across candidates per horizon, floored by replay
    for ms in (50, 250, 1000):
        per_world = []
        floors = []
        for w in worlds:
            sp = [max(r[f"w_{k}_{ms}ms"] for k in names)
                  - min(r[f"w_{k}_{ms}ms"] for k in names) for r in w["rows"]]
            fl = [abs(r[f"w_chosen_{ms}ms"] - r[f"w_replay_{ms}ms"])
                  for r in w["rows"]]
            if sp:
                per_world.append(float(np.mean(sp)))
                floors.append(float(np.percentile(fl, 90)))
        out[f"sensory_spread_{ms}ms"] = {
            "per_world_mean": per_world,
            "across_world_mean": float(np.mean(per_world)) if per_world else None,
            "replay_floor_p90_per_world": floors}
    # prediction: does imagined beat persistence; do predicted ranks match
    # realized ranks at 50 ms
    beat, agree = [], []
    for w in worlds:
        b = [np.mean([int(r["prediction_fidelity_50ms"][k]["pred_mse"]
                          < r["prediction_fidelity_50ms"][k]["persist_mse"])
                      for k in names]) for r in w["rows"]]
        g = [_rank_agreement([r["pred_judged_50ms"][k] for k in names],
                             [r[f"w_{k}_50ms"] for k in names])
             for r in w["rows"]]
        g = [x for x in g if x == x]
        if b:
            beat.append(float(np.mean(b)))
        if g:
            agree.append(float(np.mean(g)))
    out["prediction"] = {
        "imagined_beats_persistence_frac_per_world": beat,
        "predicted_vs_realized_rank_agreement_50ms_per_world": agree}
    out['training_head_beats_persistence_frac_per_world'] = [float(np.mean([
        int(r['training_prediction_fidelity_50ms'][k]['pred_mse']
            < r['training_prediction_fidelity_50ms'][k]['persist_mse'])
        for r in w['rows'] for k in names])) for w in worlds if w['rows']]
    out['action_effects_without_judge'] = {
        sense: [{'world': w['world'],
                 'training_effect_error_over_zero_effect': float(sum(
                     e['training_delta_error'] for r in w['rows'] for e in r['action_effects_50ms']
                     if e['sense'] == sense) / max(1e-30, sum(
                     e['realized_delta_mse'] for r in w['rows'] for e in r['action_effects_50ms']
                     if e['sense'] == sense))),
                 'planning_effect_error_over_zero_effect': float(sum(
                     e['planning_delta_error'] for r in w['rows'] for e in r['action_effects_50ms']
                     if e['sense'] == sense) / max(1e-30, sum(
                     e['realized_delta_mse'] for r in w['rows'] for e in r['action_effects_50ms']
                     if e['sense'] == sense)))} for w in worlds if w['rows']]
        for sense in ('tactile', 'proprioception')}
    # valuation: judge vs formula on the same REALIZED states
    corr = []
    for w in worlds:
        j = [r["valuation_50ms"][k]["judge"] for r in w["rows"] for k in names]
        f = [r["valuation_50ms"][k]["formula"] for r in w["rows"] for k in names]
        if len(j) >= 3 and np.std(j) > 0 and np.std(f) > 0:
            corr.append(float(np.corrcoef(j, f)[0, 1]))
    out["valuation_judge_vs_formula_corr_per_world"] = corr
    # optimization: chosen's realized rank among candidates; action collapse;
    # objective gradient ratio
    for ms in (50, 250, 1000):
        pw = []
        for w in worlds:
            rk = [sorted(names, key=lambda k: -r[f"w_{k}_{ms}ms"]).index("chosen")
                  for r in w["rows"]]
            if rk:
                pw.append(float(np.mean(rk)))
        out[f"chosen_realized_rank_{ms}ms_per_world"] = {
            "mean_rank_0_is_best_of_4": pw}
    diffs = [d for w in worlds for r in w["rows"]
             for d in r["candidate_exec_max_pairwise_diff"]]
    out["action_collapse"] = {
        "candidate_exec_pairwise_diff_median": float(np.median(diffs)) if diffs else None,
        "meaning": "0 means these intervention commands coincide. Controls are "
                   "rotated after execution filtering; their spread alone does "
                   "not validate action optimization or constraint preservation."}
    gr = [r["objective_grads"] for w in worlds for r in w["rows"]]
    gp = [g["goal_term_grad_norm"] for g in gr]
    gw = [g["wellness_term_grad_norm_at_w1"] for g in gr
          if g["wellness_term_grad_norm_at_w1"] is not None]
    out["objective_gradients"] = {
        "goal_term_norm_median": float(np.median(gp)) if gp else None,
        "wellness_term_norm_at_w1_median": float(np.median(gw)) if gw else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--worlds", type=int, default=3)
    ap.add_argument("--events", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument('--output', help='Write this diagnostic to a separate artifact')
    args = ap.parse_args(argv)
    if args.worlds < 1 or args.events < 1:
        ap.error('--worlds and --events must be positive')
    if free_ram_gb() < 8.0:
        print(f"REFUSED: {free_ram_gb():.1f} GB free RAM < 8.0 launch floor")
        return 2
    live, a, _s, _f = _build_model(args.ckpt, "cpu", args.seed, intention=True)
    meter, _a2, _s2, _f2 = _build_model(args.ckpt, "cpu", args.seed, intention=False)
    for mdl in (live, meter):
        if a.get("audio") == "real":
            mdl.config["modalities"].setdefault("audio", {})["data_source"] = "real"
        ape = getattr(mdl, "_agency_pay_ema", None)
        if ape is not None and torch.is_tensor(ape) and ape.numel() != 1:
            mdl._agency_pay_ema = ape.mean().reshape(1)
    limits, _ = _limits_for_run(args.ckpt, live)
    t0 = time.time()
    worlds = [run_world(args.ckpt, w, args.events, a, live, meter, limits)
              for w in range(args.worlds)]
    out = {"protocol_version": 3, "ckpt": os.path.basename(os.path.normpath(args.ckpt)),
           "ckpt_path": os.path.abspath(args.ckpt),
           "worlds": args.worlds, "events_per_world": args.events,
           "elapsed_s": time.time() - t0,
           "source_sha256": __import__("hashlib").sha256(
               open(os.path.abspath(__file__), "rb").read()).hexdigest(),
           "summary": summarize(worlds),
           "per_world": worlds,
           "note": "candidates = the planner's chosen action, an active pose "
                   "hold, and two rotated executable-command controls. Chosen "
                   "uses reach clamp + shared smoother; controls rotate that "
                   "result and are not re-clamped. This is an intervention "
                   "population, not proof that optimization is intact. "
                   "Horizons 50/250/1000 ms from restored "
                   "restored states; replay twice measures each horizon's "
                   "floor. Physics timestamps assert actual horizons. Prediction uses "
                   "pre-shift context. Raw training-head effects are measured separately "
                   "from normalized planning predictions and Judge B. Candidate population "
                   "is an offline intervention set; no training-benefit claim. "
                   "Wellness on the 0-3 bodily scale."}
    os.makedirs(os.path.join(SAVED, "consequence_localization"), exist_ok=True)
    p = args.output or os.path.join(SAVED, "consequence_localization",
                     f"{out['ckpt']}_w{args.worlds}e{args.events}_v3.json")
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tmp = p + ".tmp"
    json.dump(out, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, p)
    print(json.dumps(out["summary"], indent=1, default=float))
    print("written:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
