# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Prediction measurements with an explicit, paired persistence reference.

THE SWAP TEST (LemLite.step's paired audit; the own-command reading). On an audited model step, for each body,
the previous model step's raw senses are encoded with the command in force during the transition to this model
step in the motor row's 9 command columns; the transformer and the prediction heads predict each sense's next
token, and the prediction error is taken against the real next token. The same is computed again with the 9
command columns replaced by a control command; context, memory bias and every other sense are unchanged, and
for vision the gaze copy reads the control command's eye row. The reading per sense is 1 - (own-command errors)
/ (control errors), summed over the window's audited transitions.

Since 2026-10-02 (docs/gain_of_function_v4/agents/swap_test_phases_2026-10-02.md) there are two phases and
three control arms:
  phases  the existing block (`prediction_audit`, model steps that are multiples of AUDIT_EVERY; with
          --motor-hold 5, the 5th and last model step of each hold) and `prediction_audit_hold_first` (the hold's
          first model step, audit_blocks below);
  arms    `shuffled` (another body's ABSOLUTE command, the original control, kept beside the others for one
          segment), `size_matched` (another body's command as a change from its own pose applied to this body's
          pose, lem/model/efference.size_matched_other_order) and `shifted_own` (this body's own command plus one
          standard deviation of Gaussian noise per column, shifted_own_order). DEFINITIONS holds the text the
          stats blocks carry.
"""
import math

import torch

# THE BODIES LEFT OUT OF LEARNING-PERFORMANCE CONCLUSIONS (the operator's decision of 2026-09-25 7:47 PM PDT,
# docs/gain_of_function_v3/WORKLOG.md): body 2 has lain face down since Sept 19 and could not roll back at full
# torque in 185 attempts; until its mechanics are understood its state is excluded. The per-body block below
# reports every body and the pooled readings with and without these.
EXCLUDED_BODIES = (2,)

# THE SWAP TEST'S SPACING: one audited transition per this many model steps in each phase (the rule since
# Sep 11, commit 826f4e14a). Each audited transition costs one no-gradient forward pass per control arm.
AUDIT_EVERY = 20

# THE 2026-10-02 READINGS ON OR OFF: the hold's first model step, the size-matched and shifted-own arms, the side
# heads' size-matched control and the command-stream spread. True in every run. False makes the code compute only
# the readings it made before them; tests/test_swap_test_phases.py sets it to False to show that a learner with them
# trains bit for bit as one without them, and the cost measurement times the step both ways. Not a launch flag.
ADDED_READINGS = True

# THE CONTROL ARMS added beside the absolute roll (whose keys keep their old names: shuffled_action_mse,
# matched_prediction_mse, action_shift_mse, improvement_over_shuffled, sensitivity)
ARMS = ("size_matched", "shifted_own")

# THE SHIFTED-OWN ARM'S NOISE. One standard deviation per column (the review of Oct 2 asks for one; the August
# causal_v3 tests used half, tools/causal_test.py PERTURB_FRACTION). The standard deviation is each column's own
# over the executed-command stream: an exponential average over COMMAND_SPREAD_HORIZON_STEPS model steps of every
# body's executed command (the running-mean horizon of lem/model/efference.py CENTRE_HORIZON_STEPS and
# contact_head.CENTRE_HORIZON_STEPS: 2,000 model steps, 100 s, follows the drift of the policy and moves at most
# 1 - exp(-200/2000) = 9.5% over 200 model steps), corrected for starting at zero. The arm reads only after
# COMMAND_SPREAD_MIN_STEPS model steps of stream: at 8 bodies and a new command every 5 model steps, 200 model steps
# hold 320 independent commands, and the standard error of a standard deviation from n independent values is
# sd / sqrt(2 n) = 4% of the sd at n = 320.
SHIFTED_OWN_SIZE = 1.0
COMMAND_SPREAD_HORIZON_STEPS = 2000
COMMAND_SPREAD_MIN_STEPS = 200
# the noise's own random stream: a generator seeded with the run's seed and the model step, so no random stream
# the training draws from is touched (the efference heads' stream is run seed + 3_000_003, the contact head's
# + 4_000_004, the synergy's + 5_000_005, the liveness repair's + 6_000_006)
SHIFTED_OWN_SEED_OFFSET = 7_000_007

DEFINITIONS = {
    "version": "swap_arms_v1 (2026-10-02)",
    "sampled_at": {
        "every_20": ("model steps whose number is a multiple of 20. The trainer latches a new command on model steps "
                     "that are multiples of --motor-hold, and the command graded at model step s is the one sent "
                     "during model step s-1, so with --motor-hold 5 every graded transition is the 5th and last "
                     "model step of its command's hold, when the same command already fills the four newest "
                     "context slots"),
        "hold_first": ("the first model step of a hold: model step s with s-1 a multiple of --motor-hold (the "
                       "trainer latched the graded command at model step s-1), the first such model step at or after "
                       "each multiple of 20 (one per 20 model steps when --motor-hold is 20 or less, one per hold "
                       "otherwise); with --motor-hold 5 these are model steps 20k+1. The command appears only in the "
                       "current motor slot; the context slots hold the previous command"),
    },
    "arms": {
        "shuffled": ("another body's ABSOLUTE command (the absolute roll): body b's 26 x 9 command replaced by body "
                     "b-1's as it stands (body 0 takes the last body's). Its targets are positions and orientations "
                     "chosen for the other body's pose, so the change it asks of this body is larger than any body "
                     "was asked, and a prediction that grows with the asked change errs more with it without using "
                     "the command's effect. The original control: its keys keep their names and meaning "
                     "(shuffled_action_mse, matched_prediction_mse, improvement_over_shuffled, sensitivity) and it "
                     "is kept beside the size-matched arm for one segment, the first on this code, so the two can be "
                     "read across; then it is retired"),
        "size_matched": ("another body's command, SIZE-MATCHED: body b-1's command taken as a change from body b-1's "
                         "own pose and applied to body b's pose (lem/model/contact_head.shuffled_order, the contact "
                         "head's control: position = this pose + (other target - other pose) in cm, orientation = "
                         "(other target x inverse of the other pose) x this pose, radius and strength as they stand) "
                         "and the eyes' row as the other body's asked eye turn (its eye command minus its eyes' "
                         "azimuth and elevation at the start) added to this body's eyes' angles "
                         "(lem/model/efference.size_matched_other_order). Every asked move and turn is exactly the "
                         "other body's, so a prediction that only grows with the size of the asked change scores the "
                         "same with either command"),
        "shifted_own": ("this body's OWN command shifted: each of its 26 x 9 numbers plus Gaussian noise of one "
                        "standard deviation of that column over the executed-command stream (an exponential average "
                        "over 2,000 model steps of every body's executed command, each quaternion taken with w >= 0, "
                        "kept in LemLite._command_spread and saved in the episode companion; no gradient, never "
                        "trained), each quaternion then rescaled to length 1. The noise of model step s comes from a "
                        "generator seeded with the run's seed and s. Read only after 200 model steps of stream"),
    },
    "readings": ("improvement_over_<arm> = 1 - (sum of own-command prediction errors) / (sum of that arm's "
                 "prediction errors) over the same transitions: 0 = the arm's command predicts as well as the own "
                 "command, 1 = the own command predicts exactly where the arm's command does not, below 0 = the "
                 "arm's command predicts better. sensitivity_<arm> = ||pred(own) - pred(arm)|| / ||pred(own) - "
                 "current token||: 0 = the arm's command leaves the prediction where it was, 1 = it moves the "
                 "prediction as far as the whole predicted change"),
}


def audit_blocks(step: int, hold: int, every: int = AUDIT_EVERY) -> tuple:
    """Which swap-test blocks grade the transition of model step `step`: a tuple of "every_20" and/or "hold_first".

    THE PHASE COMES FROM THE LATCH RULE (mimo_bridge/train_mimo.py): the trainer latches a new command on model step
    L when L % hold == 0, or when no command is held (the first model step after a start or a reset), and the
    command LemLite.step grades at model step s (executed_action) is the one sent during model step s-1's burst. So
    model step s grades the FIRST model step of a hold exactly when s-1 is a latch step, and s-1 a multiple of
    `hold` is always one. Of those, the block takes the first at or after each multiple of `every`
    ((s-1) % every < hold): one transition per `every` model steps when hold <= every (a window of `hold`
    consecutive numbers holds exactly one multiple of `hold`), one per hold otherwise. With --motor-hold 5: model
    steps 20k+1. --intention-every does not enter: the planner's plan reaches the body only at a latch.
    The existing block keeps its own rule, model steps that are multiples of `every` (with --motor-hold 5, the 5th
    and last model step of each hold). Both can name one model step when hold > every."""
    out = []
    s, h, e = int(step), max(1, int(hold)), max(1, int(every))
    if s % e == 0:
        out.append("every_20")
    if s >= 1 and (s - 1) % h == 0 and (s - 1) % e < h:
        out.append("hold_first")
    return tuple(out)


def paired_errors(predicted, previous, actual, shuffled=None, size_matched=None, shifted_own=None):
    """All three representations must use the same encoder weights and raw inputs. size_matched and shifted_own:
    the predictions made with those arms' commands (ARMS), on the same transitions as `shuffled`."""
    with torch.no_grad():
        values = [x.detach().float() for x in (predicted, previous, actual)]
        if any(x.shape != values[0].shape for x in values):
            raise ValueError('prediction audit shapes differ')
        if not all(torch.isfinite(x).all() for x in values):
            return None
        p, old, target = values
        out = dict(prediction_mse=float((p-target).square().mean()),
                   persistence_mse=float((old-target).square().mean()),
                   transitions=int(p.shape[0]))
        if shuffled is not None and torch.isfinite(shuffled).all():
            s = shuffled.detach().float()
            out['shuffled_action_mse'] = float((s-target).square().mean())
            # SENSITIVITY's two halves (DESIGN.md section 3, 2026-09-22): how far swapping
            # ONLY the executed order moves the prediction, and how far the prediction
            # moves away from the present at all. Kept as mean squares so a window sums
            # them before the one square root; summarize() forms the ratio.
            out['action_shift_mse'] = float((p-s).square().mean())
            out['predicted_change_mse'] = float((p-old).square().mean())
        for arm, x in (("size_matched", size_matched), ("shifted_own", shifted_own)):
            if x is None or not torch.is_tensor(x) or x.shape != p.shape or not torch.isfinite(x).all():
                continue
            a = x.detach().float()
            out[arm + '_action_mse'] = float((a-target).square().mean())
            out[arm + '_shift_mse'] = float((p-a).square().mean())
            if 'predicted_change_mse' not in out:
                out['predicted_change_mse'] = float((p-old).square().mean())
        return out


def _arm_readings(arm, n, own, arm_mse, shift, change):
    """The pooled readings of one added arm from its window sums (n transitions)."""
    return {arm + '_transitions': int(n), arm + '_action_mse': arm_mse / n,
            'improvement_over_' + arm: (1 - own / arm_mse if arm_mse / n > 1e-12 else None),
            'sensitivity_' + arm: (math.sqrt(shift / change) if change / n > 1e-24 else None)}


def summarize(sums):
    result = {}
    for name, entry in sums.items():
        n = entry['transitions']
        p, baseline = entry['prediction_mse'], entry['persistence_mse']
        result[name] = dict(transitions=n, prediction_mse=p/n, persistence_mse=baseline/n,
                            improvement_over_persistence=(1-p/baseline if baseline/n > 1e-12 else None))
        if entry.get('shuffled_transitions', 0):
            sn = entry['shuffled_transitions']
            shuffled = entry['shuffled_action_mse']/sn
            matched = entry['matched_prediction_mse']/sn
            result[name].update(shuffled_transitions=sn, shuffled_action_mse=shuffled,
                                matched_prediction_mse=matched,
                                improvement_over_shuffled=(1-matched/shuffled if shuffled > 1e-12 else None))
            if entry.get('sensitivity_transitions', 0):
                # ||pred(real order) - pred(another body's order)|| / ||pred(real) - current||
                # over the window's audited transitions: 0 = swapping the order leaves the
                # prediction where it was (the order is ignored); 1 = swapping the order moves
                # the prediction as far as the whole predicted change; above 1 = the order
                # moves it further than the predicted change itself.
                shift = entry['action_shift_mse']
                change = entry['predicted_change_mse']
                result[name]['sensitivity'] = ((shift / change) ** 0.5 if change > 1e-24 else None)
        for arm in ARMS:
            an = entry.get(arm + '_transitions', 0)
            if an:
                result[name].update(_arm_readings(arm, an, entry[arm + '_own_prediction_mse'],
                                                  entry[arm + '_action_mse'], entry[arm + '_shift_mse'],
                                                  entry[arm + '_predicted_change_mse']))
    return dict(definition='paired_raw_tokens_v1', evaluation='online_before_update',
                scale='0 = persistence; 1 = exact next token; negative = worse than persistence',
                sensitivity_scale=('||pred(real order) - pred(another body\'s order)|| / '
                                   '||pred(real order) - current token||: 0 = the order does not '
                                   'move the prediction, 1 = it moves it as far as the whole '
                                   'predicted change'),
                limitation='Encoded senses, not physical skill or held-out action-choice proof', modalities=result,
                sampled_at=DEFINITIONS["sampled_at"]["every_20"], arms=DEFINITIONS["arms"],
                readings=DEFINITIONS["readings"], arms_version=DEFINITIONS["version"])


def paired_rows(predicted, previous, actual, shuffled=None, size_matched=None, shifted_own=None):
    """paired_errors' squared errors kept PER ROW (per body), as device tensors: nothing is read from the device
    (the trainer sums them over a report and reads them once, summarize_rows). A row with a non-finite value
    counts 0 transitions."""
    with torch.no_grad():
        p, old, target = (x.detach().float() for x in (predicted, previous, actual))
        if p.shape != old.shape or p.shape != target.shape or p.dim() < 1:
            return None

        def se(x):
            return x.reshape(x.shape[0], -1).square().mean(-1)

        def finite(x):
            return torch.isfinite(x.reshape(x.shape[0], -1)).all(-1)

        ok = finite(p) & finite(old) & finite(target)
        okf = ok.float()
        zero = torch.zeros_like(okf)
        out = dict(transitions=okf, prediction_mse=torch.where(ok, se(p - target), zero),
                   persistence_mse=torch.where(ok, se(old - target), zero))
        if shuffled is not None:
            s = shuffled.detach().float()
            if s.shape == p.shape:
                oks = ok & finite(s)
                okfs = oks.float()
                out.update(shuffled_transitions=okfs,
                           shuffled_action_mse=torch.where(oks, se(s - target), zero),
                           matched_prediction_mse=torch.where(oks, se(p - target), zero),
                           action_shift_mse=torch.where(oks, se(p - s), zero),
                           predicted_change_mse=torch.where(oks, se(p - old), zero))
        for arm, x in (("size_matched", size_matched), ("shifted_own", shifted_own)):
            if x is None or not torch.is_tensor(x):
                continue
            a = x.detach().float()
            if a.shape != p.shape:
                continue
            oka = ok & finite(a)
            out.update({arm + '_transitions': oka.float(),
                        arm + '_action_mse': torch.where(oka, se(a - target), zero),
                        arm + '_own_prediction_mse': torch.where(oka, se(p - target), zero),
                        arm + '_shift_mse': torch.where(oka, se(p - a), zero),
                        arm + '_predicted_change_mse': torch.where(oka, se(p - old), zero)})
        return out


def accumulate_rows(sums, rows):
    """Add one step's paired_rows (per sense) into the report's per-body sums (device tensors, no read). A
    sense whose body count changed starts again."""
    for name, entry in (rows or {}).items():
        if entry is None:
            continue
        target = sums.get(name)
        if target is not None and target["transitions"].shape != entry["transitions"].shape:
            target = None
        if target is None:
            sums[name] = {k: v.clone() for k, v in entry.items()}
            continue
        for key, value in entry.items():
            target[key] = target[key] + value if key in target else value.clone()


def _pooled(sums_py, keep):
    """The pooled readings over the bodies in keep, from per-body sums (lists)."""
    def tot(key):
        v = sums_py.get(key)
        return None if v is None else sum(v[b] for b in keep)
    n = tot("transitions")
    out = dict(transitions=int(n or 0))
    if n:
        pred, pers = tot("prediction_mse"), tot("persistence_mse")
        out.update(prediction_mse=pred / n, persistence_mse=pers / n,
                   ratio=(pred / pers if pers and pers / n > 1e-12 else None))
    sn = tot("shuffled_transitions")
    if sn:
        sh, matched = tot("shuffled_action_mse"), tot("matched_prediction_mse")
        shift, change = tot("action_shift_mse"), tot("predicted_change_mse")
        out.update(shuffled_transitions=int(sn),
                   improvement_over_shuffled=(1 - matched / sh if sh / sn > 1e-12 else None),
                   sensitivity=(math.sqrt(shift / change) if change / sn > 1e-24 else None))
    for arm in ARMS:
        an = tot(arm + "_transitions")
        if an:
            out.update(_arm_readings(arm, an, tot(arm + "_own_prediction_mse"), tot(arm + "_action_mse"),
                                     tot(arm + "_shift_mse"), tot(arm + "_predicted_change_mse")))
    return out


_PER_BODY_KEYS = (("prediction_mse", "transitions"), ("persistence_mse", "transitions"),
                  ("shuffled_action_mse", "shuffled_transitions"), ("matched_prediction_mse", "shuffled_transitions"),
                  ("action_shift_mse", "shuffled_transitions"), ("predicted_change_mse", "shuffled_transitions"),
                  ("size_matched_action_mse", "size_matched_transitions"),
                  ("size_matched_own_prediction_mse", "size_matched_transitions"),
                  ("shifted_own_action_mse", "shifted_own_transitions"),
                  ("shifted_own_own_prediction_mse", "shifted_own_transitions"))


def summarize_rows(sums, excluded=EXCLUDED_BODIES, phase="every_20"):
    """The report's per-body block: per sense, every body's transitions and mean squared errors (sums over the
    report divided by that body's transitions), and the pooled ratio, improvement over another body's order and
    sensitivity over all bodies and without the excluded ones (the same arithmetic as summarize). One read from
    the device per sense and key. phase: "every_20" (the existing per-body block) or "hold_first" (the
    prediction_audit_hold_first block, whose transitions are graded only on the hold's first model step)."""
    senses = {}
    for name, entry in (sums or {}).items():
        py = {k: [float(x) for x in v.tolist()] for k, v in entry.items()}
        B = len(py["transitions"])
        per = {}
        for key, nkey in _PER_BODY_KEYS:
            if key in py and nkey in py:
                per[key] = [(py[key][b] / py[nkey][b]) if py[nkey][b] > 0 else None for b in range(B)]
        per["transitions"] = [int(x) for x in py["transitions"]]
        for nkey in ("shuffled_transitions",) + tuple(arm + "_transitions" for arm in ARMS):
            if nkey in py:
                per[nkey] = [int(x) for x in py[nkey]]
        per["ratio"] = [(py["prediction_mse"][b] / py["persistence_mse"][b]) if py["persistence_mse"][b] > 0 else None
                        for b in range(B)]
        keep = [b for b in range(B) if b not in excluded]
        senses[name] = dict(per_body=per, all_bodies=_pooled(py, list(range(B))),
                            **{"without_body_" + "_".join(str(b) for b in excluded): _pooled(py, keep)})
    out = dict(definition="paired_raw_tokens_v1, per body (prediction_audit.paired_rows)",
               excluded_bodies=list(excluded),
               excluded_because="the operator's decision of 2026-09-25 7:47 PM PDT: body 2 face down since "
                                "Sept 19, excluded from learning-performance conclusions until its mechanics "
                                "are understood",
               ratio_scale="next-moment error over the error of 'the token stays the same': 1 = no better, 0 = exact",
               senses=senses)
    out.update(sampled_at=DEFINITIONS["sampled_at"][phase], arms_version=DEFINITIONS["version"])
    return out


def summarize_hold_first(sums, excluded=EXCLUDED_BODIES):
    """The `prediction_audit_hold_first` stats block: the swap test graded on the hold's first model step, every
    arm, per body and pooled with and without the excluded bodies (summarize_rows' arithmetic). Its
    improvement_over_persistence counterpart is the pooled ratio (prediction error / persistence error on those
    same transitions). Carries the arms' definitions, as the `prediction_audit` block does."""
    out = summarize_rows(sums, excluded=excluded, phase="hold_first")
    out.update(arms=DEFINITIONS["arms"], readings=DEFINITIONS["readings"])
    return out


def accumulate(sums, readings):
    for name, entry in readings.items():
        if entry is None:
            continue
        n = entry['transitions']
        target = sums.setdefault(name, dict(transitions=0, prediction_mse=0., persistence_mse=0.))
        target['transitions'] += n
        for key in ('prediction_mse', 'persistence_mse'):
            target[key] += n*entry[key]
        if 'shuffled_action_mse' in entry:
            for key, value in (('shuffled_transitions', n),
                               ('shuffled_action_mse', n*entry['shuffled_action_mse']),
                               ('matched_prediction_mse', n*entry['prediction_mse'])):
                target[key] = target.get(key, 0) + value
        if 'action_shift_mse' in entry and 'predicted_change_mse' in entry:
            for key, value in (('sensitivity_transitions', n),
                               ('action_shift_mse', n*entry['action_shift_mse']),
                               ('predicted_change_mse', n*entry['predicted_change_mse'])):
                target[key] = target.get(key, 0) + value
        for arm in ARMS:
            if arm + '_action_mse' in entry and 'predicted_change_mse' in entry:
                for key, value in ((arm + '_transitions', n),
                                   (arm + '_action_mse', n*entry[arm + '_action_mse']),
                                   (arm + '_own_prediction_mse', n*entry['prediction_mse']),
                                   (arm + '_shift_mse', n*entry[arm + '_shift_mse']),
                                   (arm + '_predicted_change_mse', n*entry['predicted_change_mse'])):
                    target[key] = target.get(key, 0) + value


# ------------------------------------------------------------------ the shifted-own arm's noise
def _aligned(cmd: torch.Tensor) -> torch.Tensor:
    """[B, H, 9] commands in float64 with each quaternion (columns 3-6, x y z w) taken with w >= 0: q and -q are one
    orientation, so a sign flip in the stream must not count as spread."""
    x = cmd.detach().to(torch.float64).clone()
    s = torch.where(x[..., 6:7] < 0, -1.0, 1.0).to(x.dtype)
    x[..., 3:7] = x[..., 3:7] * s
    return x


def command_spread_update(state, exec_t: torch.Tensor):
    """One model step of the executed-command stream into the running per-column mean and mean square (all bodies
    pooled; exponential average over COMMAND_SPREAD_HORIZON_STEPS, corrected for starting at zero). state: None or
    the dict this returns ({mean, sq: float64 [H x 9], weight, steps}); a new dict starts when the command's shape
    or device changes. No gradient."""
    with torch.no_grad():
        x = _aligned(exec_t).reshape(exec_t.shape[0], -1)
        m1, m2 = x.mean(dim=0), (x * x).mean(dim=0)
        if (not isinstance(state, dict) or not torch.is_tensor(state.get("mean"))
                or state["mean"].shape != m1.shape or state["mean"].device != m1.device):
            state = {"mean": torch.zeros_like(m1), "sq": torch.zeros_like(m1), "weight": 0.0, "steps": 0}
        a = 1.0 / float(COMMAND_SPREAD_HORIZON_STEPS)
        w_old = float(state["weight"])
        w_new = (1.0 - a) * w_old + a
        keep, take = (1.0 - a) * w_old / w_new, a / w_new
        return {"mean": state["mean"] * keep + m1 * take, "sq": state["sq"] * keep + m2 * take,
                "weight": w_new, "steps": int(state["steps"]) + 1}


def command_spread_std(state, min_steps=None):
    """[H x 9] float64: each column's standard deviation over the stream, or None before min_steps model steps
    (None = COMMAND_SPREAD_MIN_STEPS, read when called)."""
    need = COMMAND_SPREAD_MIN_STEPS if min_steps is None else int(min_steps)
    if not isinstance(state, dict) or int(state.get("steps", 0)) < need:
        return None
    return (state["sq"] - state["mean"] * state["mean"]).clamp_min(0.0).sqrt()


def shifted_own_seed(seed: int, step: int) -> int:
    """The generator seed of the shifted-own noise at model step `step` (a pure function: no state to save)."""
    return (int(seed) * 1_000_003 + SHIFTED_OWN_SEED_OFFSET + int(step)) % (2 ** 63 - 1)


def shifted_own_order(exec_t: torch.Tensor, std: torch.Tensor, step: int, seed: int,
                      size: float = SHIFTED_OWN_SIZE) -> torch.Tensor:
    """[B, H, 9] the own command plus `size` standard deviations of Gaussian noise per column (std: [H x 9] from
    command_spread_std), each quaternion rescaled to length 1 afterwards (tools/causal_test.py's perturbed copy).
    The noise is drawn on the processor from a generator seeded by shifted_own_seed, then moved to the command's
    device. Execution units in, execution units out."""
    with torch.no_grad():
        g = torch.Generator(device="cpu")
        g.manual_seed(shifted_own_seed(seed, step))
        eps = torch.randn(tuple(exec_t.shape), generator=g, dtype=torch.float32)
        sd = std.detach().to(device=exec_t.device, dtype=torch.float32).reshape(tuple(exec_t.shape[1:]))
        v = exec_t.detach().float() + float(size) * eps.to(exec_t.device) * sd
        q = v[..., 3:7]
        v = torch.cat([v[..., 0:3], q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8), v[..., 7:]], dim=-1)
        return v.to(exec_t.dtype)
