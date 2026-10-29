# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Keep every body's proprioception token following its own posture (--proprio-token-repair within_body, default off).

Version 7 (2026-09-27, lane A round 12's fix round; the record's section 47.5 and the review
proprio_token_v7_review_2026-09-27.md, finding 4, the coordinator's decision): version 6's lesson with THE HOLD'S LINE F
read on the rows the hold judges (the posture-only rows), as the MEDIAN answer of the judged bodies the lesson never
pushed, at the start AND AGAIN EACH WINDOW (300 s), so a lifted body is held at the answer the live bodies give now, not
the one they gave when the lesson started (production moved the live bodies' answer from 0.067-0.073 at 344,400 to
0.162-0.169 at 415,075); the gauges carry F in force, the step it was read and the hold's share of the steps since the
last gauge step. Everything else is version 6's.

Version 6 (2026-09-26, lane A round 10; docs/gain_of_function_v4/agents/proprio_token_2026-09-25.md sections 44-45):
version 5's lesson on the token's response per unit of the body's own posture movement, with the third review's findings
fixed: a body is pushed from the moment it reads under DEAF until it reaches LINE (the coordinator's option (b)); a
hold keeps a pushed body's fatigue answer at the live bodies' level; the gauges read the posture content with the
token's gain divided out, the fatigue answer, the jitter shares of the token and of the posture itself, and a held-out
reading of every window through the encoder saved one window earlier.

Bodies are numbered 1 to 8 in every text a person reads (the dashboard's numbering: the stats lists' index + 1);
the code's lists run 0 to 7.

WHY (record sections 42-44). After the Sept 23 deploy the proprioception encoder stopped passing the movement of whole
bodies. The mechanism (section 42): (1) the encoder's normalisations divide each vector by its own size, and for a deaf
body the part that stays constant doubled, so 2.4 to 3.6 times less of its movement passes; (2) the input carries
world-frame orientations, so the direction a body lies in is a large constant in it; (3) the next-moment lesson's
residual path pushes every token toward constant, and VICReg pushes back only for some heading groups; (4) the deploy
added pressures that reach the encoder through the core; (5) no lesson rewards a body's token following its own
posture over time. Section 44 measured the credit assignment: no other sense's prediction and no efference prediction
depends on the token, and the lessons that reach the encoder through the core move a deaf body's response by about a
thousandth of this lesson's pace. The lesson is the missing learning signal; it cuts link 5.

THE READING, per body over the window (the last WINDOW samples, one every SAMPLE_EVERY model steps: 300 s):
  movement u_b   the posture's standard deviation over the window per number, mean over its 182 numbers (each bone's
                 world quaternion and wrist-relative position);
  response s_b   the token's standard deviation over the window per number, mean over its 192 numbers (0 = the same
                 token every moment);
  ratio R_b      s_b / u_b: how far the token moves per unit of the body's own posture movement. On H84 (the lesson's
                 token, record section 42): the deaf bodies 1, 2, 6 and 8 read 1.08-1.29 and the live bodies 4, 5 and
                 7 read 4.06-4.36 at 415,075; every body but body 3 read 3.21-4.40 at 344,400, before the deploy.
  posture-only   the same window with each body's velocity (fields 7-9) and capacity (field 10) numbers held at their
                 window means: its response comes from the posture alone.

THE HINGE (version 6; the coordinator's decision of 2026-09-26, the third review's option (b)). A body is ENGAGED from
the moment its ratio reads under DEAF until it reaches LINE (one flag per body, riding runtime_state.pt); only an
engaged body is pushed, and only while under LINE. The deaf bodies are lifted to the live level; a healthy body
wandering between DEAF and LINE is left alone (at 344,400, the operator's reference, bodies 1, 2, 6, 7 and 8 read
3.21-3.63: over DEAF, under LINE; version 5 pushed them, and 17 of the 35 healthy pre-deploy readings). A released body
is engaged again only when it reads under DEAF.

THE LESSON: L = 1/2 x the sum over the engaged bodies of relu(ln LINE - ln R_b)^2, divided by the body count: a squared
hinge in log units (the reading's wander between windows that share no moment is a share of its size: 12 in 100 at every
level measured, section 43). Its gradient reaches the encoder through the POSTURE-ONLY response: the gradient used is
-sum_b relu(ln LINE - ln R_b) x grad ln s^P_b / B, the shortfall read on the real rows, the push on the posture pathway
(a push on the plain response could raise it through the velocity weights, which move 6.8 to 13.5 times further per
update than their tensor's average under Adam: section 22). A body whose posture numbers do not change at all over the
window has no ratio and is not pushed (MOVE_GUARD: a division guard, not a movement floor).

THE FATIGUE HOLD (version 6; the third review's MAJOR; the coordinator's decision of 2026-09-26 on measurement, record
section 45). THE FATIGUE ANSWER: the token's mean absolute change per number when every capacity (fatigue) input of the
window moves by FATIGUE_STEP, over its response: 0 = the token ignores fatigue; the live bodies read 0.15-0.17 at
415,075. Version 5's push raised a lifted deaf body's answer to 0.26-0.35 on movement it never trained on. Of the
review's two options, masking the lesson's gradient on the first layer's capacity columns left it at 0.25-0.41, and
keeping them (with the answer a control) was predicted to fail its own control line; masking every column constant on
the posture-only rows left up to 0.35, moving the posture weights alone up to 0.45 (proprio_mechanism/
fatigue_options.py): the rise comes through the shared layers, so the push raises the token's gain without choosing the
posture over fatigue. THE HOLD gives the lesson that choice. From a body's first push on (also after it reaches LINE;
never on a body the lesson never pushed), whenever its answer on the posture-only rows exceeds THE FATIGUE LINE F
(version 7: the median answer on the posture-only rows of the judged bodies the lesson never pushed, read at the start
and again at the first gauge step each window after the last reading, hold_line_of; with none moving, the line in force
is kept; version 6 read the largest answer on the plain rows of the bodies not engaged at the start, once), the lesson adds
1/2 relu(ln answer_b - ln F)^2 / B; the gradient used is relu(ln answer_b - ln F) x grad ln f_b / B, f_b the answer's
numerator read on the posture-only rows encoded again with every capacity input moved by FATIGUE_STEP (the push already
moves the response). The push and the hold share the lesson's per-tensor step size. With no live body at the start F is
unmeasured and the hold does not act (the gauges say so). On the lesson's path (the pace job,
proprio_token_pace_check.py) it held every pushed body at or under F on movement the path never trained on (0.09-0.16
against 0.17 at 415,075, 0.09-0.11 against 0.15 at 380,550); the deaf bodies reached LINE in 1,100 updates on both
checkpoints against 900-1,000 without it, and their gain-matched content read 0.003-0.032 above the unmoved token's
content. Its cost (proprio_mechanism/lesson_cost.py, on the CPU at 2 threads, times under the machine's load): the graph
a pushing step keeps for its backward, 89 MB without the hold and 134 MB with every pushed body held; after the lift the
hold's reading keeps no graph (0.137 s a step against 0.079 s for a lesson with nothing to do); the card's times are a
qualification job. Version 7 reads every moving body's posture-only answer on a gauge step (every 100th model step: one
more encoding of the window's rows without a graph), which F's reading and the watcher's printed readings need.

THE LINES (LINE, DEAF): the calibration's RULE LINE (the live bodies' level: the median ratio of the pre-deploy windows
that share no moment) and RULE DEAF (the tolerance bound under which a body reads unlike the pre-deploy bodies), both in
the lesson's units (the lines file's LINE and DEAF). UNMEASURED here: the lesson refuses to run without them
(--proprio-token-line, --proprio-token-deaf); a value outside [LINE_MIN, LINE_MAX] is a mistyping, and DEAF must lie
under LINE.

SKIP (--proprio-token-skip): the bodies, numbered from 1, the lesson never pushes (body 3 on H84: face down since Sept
19, judged by no line).

ITS STRENGTH (unchanged from version 4). The lesson's gradient is computed on its own and added AFTER production's
clipping, each encoder tensor's share rescaled to STRENGTH times that tensor's running gradient size (the root mean
square over the tensor of the square root of Adam's bias-corrected second moment).

THE GAUGES. Every step: each body's ratio, response, movement, jitter, the JITTER SHARES of the token and of the posture
itself (jitter_share: one formula for the lesson, the calibration and the trace; the watcher's K2 judges their
difference, which production's own change of behaviour moves in both at once), the log shortfall and the engaged flags.
Every GAUGE_EVERY-th model step (the stats record's cadence) and at the start, also: ratio_posture_only; content, the
version-4 posture content at CONTROL_LAMBDA; content_gain_matched, the same with the token's gain since the start
divided out (the content at CONTROL_LAMBDA x G^2, G = the ratio now over the start's: a pure gain reads the start's
content, a lift that adds movement without posture information reads less: the third review's planted token);
fatigue_answer, the token's mean absolute change per number when every capacity input moves by FATIGUE_STEP, over its
response; lesson_pace and attributed (the lesson's own first-order change of each body's ln ratio, per update and summed
since the start); the start (the first full window's readings, before the first update, on the real input); HELD_OUT: at
the first gauge step at least one window (FILL model steps) after the encoder was last saved, the current window, all of
it newer than that encoder, read through it (its ratio and gain-matched content): a reading of the lift on movement the
encoder never trained on; the encoder is then saved again (it is 312 x 192 plus three 192 x 192 layers). The start, the
attributed lift, the engaged and pushed flags, the saved encoder and the last held-out reading ride runtime_state.pt.

Nothing changes at the switch: no weight, no token, no stored state; the lesson adds nothing to the loss (it is applied
after the clip) and nothing at all until the window is full. With the flag off LemLite imports nothing from here:
today's model.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import torch

VERSION = "proprio_token_repair_v7"
# the window's rows are the same raw inputs in every version from 3 on (versions 1 and 2 kept 200 samples)
READS_VERSIONS = ("proprio_token_repair_v1", "proprio_token_repair_v2", "proprio_token_repair_v3",
                  "proprio_token_repair_v4", "proprio_token_repair_v5", "proprio_token_repair_v6", VERSION)
MODES = ("off", "within_body")

# THE WINDOW: 300 s of body time, the replay ring's span (enable_replay's 6,000 moments), at the collapse watch's own
# sampling (lem_model.py: a sample when step_count % 5 == 0): 1,200 samples.
SAMPLE_EVERY = 5
WINDOW = 1200
# The lesson reads only a full window (review finding 5: no reading on a length nothing was measured on).
MIN_SAMPLES = WINDOW
# THE LINES on the ratio: unmeasured here (module docstring).
LINE: Optional[float] = None
DEAF: Optional[float] = None
# A TYPING GUARD on the lines, not a measurement: the calibration reads ratios of 0.27-5.9 on H84's 22 checkpoints and
# version 4's line was an R2 of 0.32; a value outside [LINE_MIN, LINE_MAX] is a mistyping. K1 checks the values against
# the lines file.
LINE_MIN, LINE_MAX = 1.0, 20.0
# THE STRENGTH: the lesson's gradient on each encoder tensor, as a share of that tensor's running gradient size
# (the tensor-average step in learning rates; per weight see the module docstring).
STRENGTH = 0.02
STRENGTH_MAX = 0.1
# A DIVISION GUARD, not a movement floor: a body whose posture numbers do not change over the window (to float32's
# resolution of numbers of size 0.1-1: 1e-7 relative) has no ratio. Every H84 body moved 0.061-0.254 per number.
MOVE_GUARD = 1e-6
# The log of a response never meets 0: the smallest response any H84 body read is 0.032 per number (body 3).
RESPONSE_EPS = 1e-8
# The gauges' cadence: the stats record's (every 100 model steps; the watcher's REPORT_EVERY).
GAUGE_EVERY = 100
# THE CONTENT CONTROL's decoder (version 4's reading): ten blocks, 50 moments removed each side of a held-out block,
# the penalty the version-4 calibration chose by its RULE LAMBDA (200, record section 41; test pinned); the control
# shifted 3,000 moments. Read on gauge steps only, without gradient.
FOLDS = 10
GAP = 10
SHIFT = 600
CONTROL_LAMBDA = 200.0
# THE FATIGUE ANSWER's capacity step: 0.042, the 95th percentile of a body's mean capacity's 300-s change on segments 32
# and 33 (record section 32), the step the third review read the answer at and the hold's rows move by.
FATIGUE_STEP = 0.042
N_BONE_FIELDS = 12
POSTURE_FIELDS = tuple(range(7))       # quaternion (0-3) and wrist-relative position / 200 cm (4-6)
VELOCITY_FIELDS = (7, 8, 9)            # each bone's velocity / 500 cm/s
CAPACITY_FIELD = 10                    # the capacity (fatigue) field of each bone row
HELD_FIELDS = VELOCITY_FIELDS + (CAPACITY_FIELD,)
INPUT_FIELDS = {"quaternion": (0, 1, 2, 3), "position": (4, 5, 6), "velocity": (7, 8, 9), "capacity": (10,),
                "row_index": (11,)}
_EPS = 1e-6


def _columns(n_inputs: int, fields) -> torch.Tensor:
    if n_inputs % N_BONE_FIELDS:
        raise ValueError("the proprioceptive input is %d numbers, not whole %d-number bone rows"
                         % (n_inputs, N_BONE_FIELDS))
    return torch.tensor([j for j in range(n_inputs) if j % N_BONE_FIELDS in fields], dtype=torch.long)


def posture_columns(n_inputs: int) -> torch.Tensor:
    """The posture part's columns in the flat proprioceptive input: fields 0-6 of every 12-number bone row."""
    return _columns(n_inputs, POSTURE_FIELDS)


def capacity_columns(n_inputs: int) -> torch.Tensor:
    """The capacity (fatigue) field's columns in the flat proprioceptive input: field 10 of every bone row."""
    return _columns(n_inputs, (CAPACITY_FIELD,))


def held_columns(n_inputs: int) -> torch.Tensor:
    """The columns the posture-only rows hold at their window mean: velocity (7-9) and capacity (10) of every bone."""
    return _columns(n_inputs, HELD_FIELDS)


def capacity_held(X: torch.Tensor) -> torch.Tensor:
    """X [B, n, F] with each body's capacity fields held at their window mean (a copy)."""
    cols = capacity_columns(X.shape[-1]).to(X.device)
    out = X.clone()
    out[..., cols] = X[..., cols].mean(dim=1, keepdim=True)
    return out


def posture_only(X: torch.Tensor) -> torch.Tensor:
    """X [B, n, F] with each body's velocity and capacity fields held at their window mean (a copy): rows whose only
    change over the window is the posture's (quaternion and position), with the row index as it is."""
    cols = held_columns(X.shape[-1]).to(X.device)
    out = X.clone()
    out[..., cols] = X[..., cols].mean(dim=1, keepdim=True)
    return out


def response(Z: torch.Tensor) -> torch.Tensor:
    """Z [B, n, D] -> [B]: the standard deviation over the window per number, mean over numbers (float32; the
    gradient flows)."""
    return Z.float().std(dim=1).mean(dim=-1)


def movement(P: torch.Tensor) -> torch.Tensor:
    """P [B, n, K] posture -> [B]: the posture's standard deviation over the window per number, mean over numbers."""
    return P.detach().float().std(dim=1).mean(dim=-1)


def jitter_share(x: torch.Tensor) -> torch.Tensor:
    """x [B, n, K] -> [B]: THE JITTER SHARE of a signal over the window, the one formula of the lesson's gauges, the
    calibration and the trace (docs/gain_of_function_v4/proprio_mechanism/jitter_share.py): the mean over its numbers
    of (the mean squared change between consecutive samples) / (2 x its variance over the window), which is 1 minus the
    lag-one autocorrelation at 0.25 s: 0 = smooth, 1 = consecutive samples unrelated. A gain leaves it as it was."""
    x = x.detach().float()
    d = (x[:, 1:] - x[:, :-1]).square().mean(1)
    v = x.var(1).clamp_min(1e-12)
    return (d / (2.0 * v)).mean(-1)


def ratio_of(s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The response per unit movement, and 0 where the body did not move (u <= MOVE_GUARD: no ratio)."""
    return torch.where(u > MOVE_GUARD, s / u.clamp_min(MOVE_GUARD), torch.zeros_like(s))


def log_shortfall(ratio: torch.Tensor, u: torch.Tensor, line: float, pushable: torch.Tensor) -> torch.Tensor:
    """relu(ln line - ln ratio) for every body that moved and may be pushed; 0 otherwise (floats, no gradient)."""
    r = ratio.detach().float()
    has = (u > MOVE_GUARD) & pushable
    short = torch.relu(math.log(line) - torch.log(r.clamp_min(RESPONSE_EPS)))
    return torch.where(has, short, torch.zeros_like(short))


def posture_content(Z: torch.Tensor, P: torch.Tensor, lam, folds: Optional[int] = None,
                    gap: Optional[int] = None, control: bool = False) -> torch.Tensor:
    """Version 4's reading, kept as the content control and for the calibration's record: the review's whole-ring
    reading of each body's window: Z [B, n, D] tokens (gradient flows through it), P [B, n, K] posture (data), both in
    time order. The window is cut into `folds` blocks (FOLDS); each is predicted by a ridge decoder fitted on the rest
    with `gap` samples (GAP) removed each side of it, on tokens centred on the fitted part, penalty lam x identity in
    token units (one number, or one per body: [B]); R2 = 1 - SSE / SST over every held-out sample and posture number,
    SST about the window's mean posture. control=True rolls the tokens SHIFT samples (at most half the window) against
    the posture. Returns R2 [B] (float32)."""
    folds = FOLDS if folds is None else int(folds)
    gap = GAP if gap is None else int(gap)
    Z = Z.float()
    P = P.detach().float()
    B, n, D = Z.shape
    if n < 2 * folds:
        raise ValueError("a window of %d samples cannot hold %d blocks" % (n, folds))
    lam_t = torch.as_tensor(lam, dtype=Z.dtype, device=Z.device)
    if lam_t.dim() and lam_t.numel() not in (1, B):
        raise ValueError("the penalty is one number or one per body, not %d for %d bodies" % (lam_t.numel(), B))
    if control:
        Z = torch.roll(Z, shifts=min(SHIFT, n // 2), dims=1)
    edges = torch.linspace(0, n, folds + 1).long().tolist()
    eye = torch.eye(D, dtype=Z.dtype, device=Z.device)
    pen = (lam_t.reshape(-1, 1, 1) if lam_t.dim() else lam_t) * eye
    sse = Z.new_zeros(B)
    for f in range(folds):
        a, b = edges[f], edges[f + 1]
        keep = torch.ones(n, dtype=torch.bool, device=Z.device)
        keep[max(0, a - gap):min(n, b + gap)] = False
        Zt, Pt = Z[:, keep], P[:, keep]
        mz, mp = Zt.mean(1, keepdim=True), Pt.mean(1, keepdim=True)
        Zc = Zt - mz
        # the penalised Gram matrix is symmetric positive definite: Cholesky (the batched float32 LU solve failed on
        # this machine's MKL at 2 threads: test_the_content_control_solves_eight_bodies_at_the_jobs_two_threads)
        W = torch.cholesky_solve(Zc.transpose(1, 2) @ (Pt - mp), torch.linalg.cholesky(Zc.transpose(1, 2) @ Zc + pen))
        sse = sse + (P[:, a:b] - ((Z[:, a:b] - mz) @ W + mp)).square().sum((1, 2))
    sst = (P - P.mean(1, keepdim=True)).square().sum((1, 2)).clamp_min(_EPS)
    return 1.0 - sse / sst


def gain_matched_content(Z: torch.Tensor, P: torch.Tensor, gain, lam: float = CONTROL_LAMBDA) -> torch.Tensor:
    """[B]: the posture content with the token's gain divided out: posture_content at the penalty lam x G^2 per body,
    G = gain [B] (the ratio now over the ratio at the reference). A token scaled by G reads at the penalty lam what the
    unscaled token reads at lam / G^2 (the ridge penalises the decoder's weights, which shrink by G), so this reads the
    information the unscaled token carries: a pure gain reads the reference's content; a lift that adds movement
    carrying no posture information reads less (the third review's planted token: 0.019-0.060 against a start of
    0.21-0.24). A gain that is not a positive finite number is taken as 1."""
    g = torch.as_tensor(gain, dtype=torch.float32, device=Z.device).reshape(-1)
    g = torch.where(torch.isfinite(g) & (g > 0), g, torch.ones_like(g))
    return posture_content(Z, P, lam=float(lam) * g.square())


def fatigue_stepped(X: torch.Tensor) -> torch.Tensor:
    """X [..., F] with every capacity (fatigue) input moved by FATIGUE_STEP (a copy)."""
    cols = capacity_columns(X.shape[-1]).to(X.device)
    out = X.clone()
    out[..., cols] = out[..., cols] + FATIGUE_STEP
    return out


def fatigue_change(Z: torch.Tensor, Zf: torch.Tensor) -> torch.Tensor:
    """[B]: the token's mean absolute change per number between Z and Zf [B, n, D], the same rows encoded before and
    after the fatigue step (the fatigue answer's numerator; the gradient flows)."""
    return (Zf.float() - Z.float()).abs().mean(dim=(1, 2))


def fatigue_answer(encode, X: torch.Tensor, Z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """[B]: the FATIGUE ANSWER (the third review's reading): the token's mean absolute change per number when every
    capacity (fatigue) input of the window moves by FATIGUE_STEP, over the token's response s: 0 = the token ignores
    fatigue. Z [B, n, D] is encode's token of X [B, n, F]. No gradient."""
    B, n, F = X.shape
    with torch.no_grad():
        z2 = encode(fatigue_stepped(X).reshape(B * n, F)).reshape(B, n, -1)
        return fatigue_change(Z.detach(), z2) / s.detach().float().clamp_min(RESPONSE_EPS)


def fatigue_line_of(answers, live) -> Optional[float]:
    """The live bodies' LARGEST fatigue answer (live [B] bools): the judgment's declared line (rule LS's fatigue line on
    the plain rows at update 0; the watcher's A1 (vi) at the start); version 6's hold read its line so. None when no
    body is live."""
    vals = [float(a) for a, ok in zip(answers or [], live or []) if ok and a is not None and math.isfinite(float(a))]
    return max(vals) if vals else None


def hold_line_of(answers_posture_only, never_pushed) -> Optional[float]:
    """THE HOLD'S LINE F (version 7; module docstring, THE FATIGUE HOLD): the MEDIAN fatigue answer on the posture-only
    rows (the rows the hold judges) of the judged bodies the lesson never pushed (never_pushed [B] bools); None when no
    such body moved: at the start the hold then does not act, and at a later window the line in force is kept."""
    vals = sorted(float(a) for a, ok in zip(answers_posture_only or [], never_pushed or [])
                  if ok and a is not None and math.isfinite(float(a)))
    if not vals:
        return None
    k = len(vals) // 2
    return vals[k] if len(vals) % 2 else 0.5 * (vals[k - 1] + vals[k])


def running_size(optimizer, p: torch.Tensor, beta2: float) -> Optional[torch.Tensor]:
    """The size of the gradient `p` receives: the square root of Adam's bias-corrected second moment, or None
    before its first update."""
    st = optimizer.state.get(p) or {}
    v, step = st.get("exp_avg_sq"), st.get("step")
    if v is None or step is None:
        return None
    s = float(step.item()) if torch.is_tensor(step) else float(step)
    if s < 1:
        return None
    return (v / (1.0 - beta2 ** s)).sqrt()


def _rms(t: torch.Tensor) -> float:
    return float(t.detach().float().square().mean().sqrt())


def _line(value, name: str) -> float:
    if value is None:
        raise ValueError("the %s on the response per unit movement is unmeasured: the calibration job "
                         "(docs/gain_of_function_v4/proprio_token_measurements.py calibrate, RULE %s) measures it; "
                         "give the lines file's value with --proprio-token-%s" % (name.lower(), name, name.lower()))
    if not math.isfinite(float(value)):
        raise ValueError("--proprio-token-%s must be a finite ratio, not %r" % (name.lower(), value))
    if not LINE_MIN <= float(value) <= LINE_MAX:
        raise ValueError("--proprio-token-%s %r lies outside [%g, %g]: the line is a response per unit of posture "
                         "movement (H84's live bodies 3.2-5.9), so this is a mistyping (version 4's line was an R2)"
                         % (name.lower(), value, LINE_MIN, LINE_MAX))
    return float(value)


def parse_skip(skip) -> tuple:
    """--proprio-token-skip as a sorted tuple of body numbers from 1 ("3", "3,5", [3], None -> (3,), (3, 5), (3,),
    ()). Refuses anything that is not a positive whole number."""
    if skip is None:
        return ()
    items = skip.replace(",", " ").split() if isinstance(skip, str) else list(skip)
    out = set()
    for v in items:
        try:
            n = int(str(v).strip())
        except ValueError:
            raise ValueError("--proprio-token-skip names bodies by number from 1, not %r" % (v,))
        if n < 1 or str(n) != str(v).strip():
            raise ValueError("--proprio-token-skip names bodies by number from 1, not %r" % (v,))
        out.add(n)
    return tuple(sorted(out))


class WithinBody:
    """The window of raw proprioceptive inputs per body, the response-per-unit-movement reading, the hinge (engaged
    under DEAF, released at LINE), the hold, the lesson and its gauges. PUBLIC on the model (LemLite.proprio_within):
    the episode companion never carries it; the window, the start, the attributed lift, the engaged and pushed flags
    and the saved encoder ride runtime_state.pt under "proprio_token_repair" (state_dict / load_state_dict)."""

    def __init__(self, line: Optional[float] = None, strength: Optional[float] = None, skip=None,
                 deaf: Optional[float] = None):
        self.line = _line(LINE if line is None else line, "LINE")
        self.deaf = _line(DEAF if deaf is None else deaf, "DEAF")
        if not self.deaf < self.line:
            raise ValueError("--proprio-token-deaf %r must lie under --proprio-token-line %r: a body is engaged under "
                             "DEAF and released at LINE" % (self.deaf, self.line))
        strength = STRENGTH if strength is None else strength
        if not (math.isfinite(float(strength)) and 0.0 < float(strength) <= STRENGTH_MAX):
            raise ValueError("--proprio-token-strength must be above 0 and at most %g, not %r"
                             % (STRENGTH_MAX, strength))
        self.strength = float(strength)
        self.skip = parse_skip(skip)
        # rows_transform: the rows the encoder reads when --proprio-heading free turns them (review P2-5): the
        # posture whose movement the ratio divides by is the posture in those rows
        self.rows_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self.ring: Optional[torch.Tensor] = None       # [B, WINDOW, F], written circularly
        self.count = 0                                 # samples written (saturates at WINDOW)
        self.ptr = 0                                   # next slot to write
        self.cols: Optional[torch.Tensor] = None
        self.last: Dict[str, object] = {}
        self.applied: Dict[str, object] = {}
        self.restored_from: Optional[str] = None
        self.start: Optional[dict] = None              # the first full window's readings, before the first update
        self.attributed: Optional[List[float]] = None  # per body: the lesson's summed first-order change of ln R_b
        self.engaged: Optional[List[bool]] = None      # per body: under DEAF once, not yet back at LINE
        self.pushed_ever: Optional[List[bool]] = None  # per body: pushed at least once since the start (the hold's)
        self.snapshot: Optional[Dict[str, torch.Tensor]] = None   # the encoder saved for the held-out reading
        self.snapshot_step: Optional[int] = None
        self.held_out: Optional[dict] = None           # the last held-out reading
        self.fatigue_line: Optional[float] = None      # THE HOLD'S LINE F in force (version 7: read again each window)
        self.fatigue_line_step: Optional[int] = None   # the step F was last read
        self._hold_counts: List[int] = []              # per body: steps since the last gauge step the hold acted on it
        self._hold_any = 0                             # steps since the last gauge step the hold acted on any body
        self._hold_steps = 0                           # steps since the last gauge step with a full window
        self._applied_since = 0                        # lesson updates since the last attribution
        self._step: Optional[int] = None               # the model step observe() last saw
        self._due = False                              # this step's lesson reads the gauges
        self._read_graph: Optional[torch.Tensor] = None  # this step's ln s_b with its graph (the attribution)

    def pushable(self, B: int, device=None) -> torch.Tensor:
        """[B] bools: the bodies the lesson may push (all but SKIP's)."""
        return torch.tensor([(b + 1) not in self.skip for b in range(B)], dtype=torch.bool, device=device)

    # ---------------------------------------------------------------- the window
    def reset(self) -> None:
        """Forget the window (an episode reset or a body rebuild moves the body discontinuously). The start, the
        attributed lift, the engaged and pushed flags and the saved encoder describe the deploy and are kept."""
        self.ring = None
        self.count = 0
        self.ptr = 0

    def observe(self, x: torch.Tensor, step: int, real: bool) -> bool:
        """Keep this moment's raw input [B, F] when the step is a sampling step and the reading is real. Returns
        whether it was kept. Every call records the step (the gauges' cadence)."""
        self._step = int(step)
        if not real or x is None or int(step) % SAMPLE_EVERY != 0:
            return False
        x = x.detach().reshape(x.shape[0], -1).float()
        B, F = x.shape
        if self.ring is None or tuple(self.ring.shape) != (B, WINDOW, F) or self.ring.device != x.device:
            if self.ring is not None and self.count:
                self.last["ring_restarted"] = "body count, input width or device changed"
            self.ring = torch.zeros(B, WINDOW, F, device=x.device)
            self.count = 0
            self.ptr = 0
            self.cols = None
        self.ring[:, self.ptr] = x
        self.ptr = (self.ptr + 1) % WINDOW
        self.count = min(self.count + 1, WINDOW)
        return True

    def window(self) -> Optional[torch.Tensor]:
        """The kept samples in time order, [B, n, F], or None below MIN_SAMPLES."""
        if self.ring is None or self.count < MIN_SAMPLES:
            return None
        if self.count < WINDOW:
            return self.ring[:, :self.count]
        return torch.cat([self.ring[:, self.ptr:], self.ring[:, :self.ptr]], dim=1)

    # ---------------------------------------------------------------- the hinge
    def engage(self, ratio: torch.Tensor, has: torch.Tensor) -> torch.Tensor:
        """[B] bools: the engaged bodies after this reading (the hinge, module docstring). A body with no reading
        (skipped, or a posture that did not change) is released."""
        B = int(ratio.shape[0])
        r = [float(v) for v in ratio.detach()]
        h = [bool(v) for v in has]
        if self.engaged is None or len(self.engaged) != B:
            self.engaged = [h[b] and r[b] < self.deaf for b in range(B)]
        else:
            for b in range(B):
                if self.engaged[b] and (not h[b] or r[b] >= self.line):
                    self.engaged[b] = False
                elif not self.engaged[b] and h[b] and r[b] < self.deaf:
                    self.engaged[b] = True
        return torch.tensor(self.engaged, dtype=torch.bool, device=ratio.device)

    # ---------------------------------------------------------------- the reading and the lesson
    def posture(self, X: torch.Tensor) -> torch.Tensor:
        """The posture part of the rows the encoder reads, [B, n, 182]."""
        if self.cols is None or self.cols.device != X.device:
            self.cols = posture_columns(X.shape[-1]).to(X.device)
        rows = X if self.rows_transform is None else self.rows_transform(X)
        return rows.index_select(-1, self.cols)

    def lesson(self, encode, encode_with=None, weights_now=None) -> Optional[torch.Tensor]:
        """encode(x [N, F]) -> tokens [N, D] through the live encoder and the output LayerNorm, with gradient;
        encode_with(weights) -> such a function through saved encoder weights and weights_now() -> the encoder's
        weights now (both optional: the held-out reading needs them). Returns the lesson (a scalar whose gradient is the
        push, -sum_b shortfall_b x grad ln s^P_b / B over the engaged bodies, plus the hold, sum_b over_b x grad ln f_b
        / B over the pushed bodies whose fatigue answer is over F) or None when the window is not full or neither acts.
        The lesson is never added to the loss: apply() adds its gradient after production's clipping."""
        carried = {k: self.last[k] for k in ("ratio_posture_only", "spread_posture_only", "content",
                                             "content_control", "content_gain_matched", "fatigue_answer",
                                             "lesson_pace", "attributed", "gauge_step", "ring_restarted",
                                             "start_retaken", "hold_share_since_gauge", "hold_any_share_since_gauge",
                                             "hold_steps_since_gauge") if k in self.last}
        X = self.window()
        self._read_graph = None
        self._due = False
        if X is None:
            self.last = dict(carried, samples=int(self.count), lesson=None)
            return None
        B, n, F = X.shape
        if self.start is not None and len(self.start.get("ratio") or []) != B:
            # a rebuild with another body count: the old start describes other bodies
            carried["start_retaken"] = "the body count changed from %d to %d" % (len(self.start["ratio"]), B)
            self.start, self.attributed, self.engaged, self.pushed_ever = None, None, None, None
        due = self.start is None or (self._step is not None and self._step % GAUGE_EVERY == 0)
        P = self.posture(X)
        u = movement(P)
        pushable = self.pushable(B, X.device)
        # the plain token (the real rows, what the core receives): its graph is kept on a gauge step only (each body's
        # gradient of ln s_b is the attribution's reading gradient)
        with torch.set_grad_enabled(due and torch.is_grad_enabled()):
            Z = encode(X.reshape(B * n, F)).reshape(B, n, -1)
            with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                s_graph = response(Z)
        s = s_graph.detach()
        ratio = ratio_of(s, u)
        has = (u > MOVE_GUARD) & pushable
        eng = self.engage(ratio, has)
        short = log_shortfall(ratio, u, self.line, pushable & eng)
        pushed = short > 0
        if self.pushed_ever is None or len(self.pushed_ever) != B:
            self.pushed_ever = [False] * B
        self.pushed_ever = [bool(pe) or bool(p) for pe, p in zip(self.pushed_ever, pushed)]
        # THE HOLD's line in force (version 7: read at the start and again each window, the median posture-only answer
        # of the judged bodies never pushed); a start set by hand without the lesson's own reading (the time and cost
        # measurements set one) gives it
        f_line = self.fatigue_line if self.fatigue_line is not None else (
            None if self.start is None else self.start.get("fatigue_line"))
        # THE HOLD's candidates: every body pushed since the start that moved, once F is measured
        if f_line:
            cand = torch.tensor(self.pushed_ever, dtype=torch.bool, device=X.device) & has
        else:
            cand = torch.zeros(B, dtype=torch.bool, device=X.device)
        lesson = None
        sp = None
        answer_p = [None] * B
        over = torch.zeros(B, device=X.device)
        grad_on = torch.is_grad_enabled()
        if bool(pushed.any()) or bool(cand.any()) or due:
            Xp = posture_only(X)
            terms = []
            # the push needs the posture-only response's graph; the hold's reading does not
            with torch.set_grad_enabled(bool(pushed.any()) and grad_on):
                Zp = encode(Xp.reshape(B * n, F)).reshape(B, n, -1)
                with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                    sp_graph = response(Zp)
                    if bool(pushed.any()):
                        terms.append(-(short * torch.log(sp_graph.clamp_min(RESPONSE_EPS))).sum() / B)
            sp = sp_graph.detach()
            # THE HOLD's reading, without a graph, on the posture-only rows: each candidate's answer, and on a gauge
            # step every moving body's (F's reading at the start and each window; the live bodies' answer now)
            read = has if due else cand
            if bool(read.any()):
                idx = read.nonzero().flatten()
                with torch.no_grad():
                    Zpf = encode(fatigue_stepped(Xp.index_select(0, idx)).reshape(-1, F)).reshape(len(idx), n, -1)
                    with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                        ans = (fatigue_change(Zp.detach().index_select(0, idx), Zpf)
                               / sp.float().index_select(0, idx).clamp_min(RESPONSE_EPS))
                for j, b in enumerate(idx.tolist()):
                    answer_p[b] = float(ans[j])
            if bool(cand.any()):
                idx = cand.nonzero().flatten()
                with torch.no_grad():
                    ans_c = torch.tensor([answer_p[b] for b in idx.tolist()], device=X.device)
                    ov = torch.relu(torch.log(ans_c.clamp_min(RESPONSE_EPS)) - math.log(f_line))
                for j, b in enumerate(idx.tolist()):
                    over[b] = float(ov[j])
                hot = idx[ov > 0]
                if hot.numel() and grad_on:
                    # the hold's graph, built only for the candidates over the line (after the lift the answers sit
                    # under it, and a step then encodes the two sets of rows once each, without a graph)
                    with torch.enable_grad():
                        xo = Xp.index_select(0, hot)
                        # while the push runs, its posture-only graph serves the hold too
                        zo = (Zp.index_select(0, hot) if Zp.requires_grad
                              else encode(xo.reshape(-1, F)).reshape(len(hot), n, -1))
                        zof = encode(fatigue_stepped(xo).reshape(-1, F)).reshape(len(hot), n, -1)
                        with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                            f_graph = fatigue_change(zo, zof)
                            terms.append((over.index_select(0, hot) * torch.log(f_graph.clamp_min(RESPONSE_EPS))).sum()
                                         / B)
            if terms:
                lesson = terms[0] if len(terms) == 1 else terms[0] + terms[1]
        with torch.no_grad():
            zd = Z.detach().float()
            half = 0.5 * float(short.square().sum()) / B
            held_now = [bool(v > 0) for v in over]
            # the hold's share of the steps since the last gauge step (the watcher prints it: the review's finding 4)
            if len(self._hold_counts) != B:
                self._hold_counts = [0] * B
            self._hold_counts = [c + int(h) for c, h in zip(self._hold_counts, held_now)]
            self._hold_any += int(any(held_now))
            self._hold_steps += 1
            last = dict(carried, samples=int(n), lesson=half, hold=0.5 * float(over.square().sum()) / B,
                        spread=[float(v) for v in s],
                        posture_spread=[float(v) for v in u], ratio=[float(v) for v in ratio],
                        shortfall=[float(v) for v in short],
                        # the token's mean squared change between consecutive samples, per number
                        jitter=[float(v) for v in (zd[:, 1:] - zd[:, :-1]).square().mean(dim=(1, 2))],
                        # the jitter shares of the token and of the posture itself (the watcher's K2 judges the first
                        # minus the second: a change of behaviour moves both, a noise-like token only the first)
                        jitter_share=[float(v) for v in jitter_share(zd)],
                        posture_jitter_share=[float(v) for v in jitter_share(P)],
                        below_line=int(pushed.sum()), pushed=[bool(v) for v in pushed],
                        engaged=list(self.engaged), pushed_ever=list(self.pushed_ever),
                        fatigue_line=f_line, fatigue_answer_posture_only=answer_p,
                        held=held_now, no_movement=int((u <= MOVE_GUARD).sum()),
                        skipped=list(self.skip))
            if due:
                last["spread_posture_only"] = [float(v) for v in sp]
                last["ratio_posture_only"] = [float(v) for v in ratio_of(sp, u)]
                with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                    last["content"] = [float(v) for v in posture_content(zd, P, lam=CONTROL_LAMBDA)]
                    last["content_control"] = [float(v) for v in posture_content(zd, P, lam=CONTROL_LAMBDA,
                                                                                  control=True)]
                    ref = (self.start or {}).get("ratio")
                    gain = ([float(r) / float(r0) if r0 and r0 > 0 else 1.0 for r, r0 in zip(ratio, ref)]
                            if ref else [1.0] * B)
                    last["content_gain_matched"] = [float(v) for v in gain_matched_content(zd, P, gain)]
                    last["fatigue_answer"] = [float(v) for v in fatigue_answer(encode, X, zd, s)]
                last["gauge_step"] = self._step
                steps_seen = max(1, self._hold_steps)
                last["hold_share_since_gauge"] = [c / steps_seen for c in self._hold_counts]
                last["hold_any_share_since_gauge"] = self._hold_any / steps_seen
                last["hold_steps_since_gauge"] = self._hold_steps
                self._hold_counts, self._hold_any, self._hold_steps = [0] * B, 0, 0
            if self.start is None:
                live = [bool(h) and not e for h, e in zip(has, self.engaged)]
                # THE HOLD'S LINE at the start (version 7): the median posture-only answer of the live bodies (the
                # judged bodies that moved and were not engaged, so never pushed)
                f0 = hold_line_of(answer_p, live)
                self.start = dict(step=self._step, ratio=last["ratio"], ratio_posture_only=last["ratio_posture_only"],
                                  spread=last["spread"], spread_posture_only=last["spread_posture_only"],
                                  posture_spread=last["posture_spread"], jitter=last["jitter"],
                                  jitter_share=last["jitter_share"], posture_jitter_share=last["posture_jitter_share"],
                                  content=last["content"], content_control=last["content_control"],
                                  content_gain_matched=last["content_gain_matched"],
                                  fatigue_answer=last["fatigue_answer"], fatigue_answer_posture_only=list(answer_p),
                                  shortfall=last["shortfall"], pushed=last["pushed"], engaged=list(self.engaged),
                                  live=live, fatigue_line=f0, line=self.line, deaf=self.deaf, skip=list(self.skip))
                self.fatigue_line, self.fatigue_line_step = f0, self._step
                self.attributed = [0.0] * B
                self._applied_since = 0
            elif (due and self.fatigue_line_step is not None and self._step is not None
                  and int(self._step) - int(self.fatigue_line_step) >= WINDOW * SAMPLE_EVERY):
                # THE HOLD'S LINE READ AGAIN each window (the review's finding 4, the coordinator's decision): the
                # median posture-only answer of the judged bodies the lesson never pushed, as they answer NOW (a
                # lifted body is held at the live bodies' answer now, not at theirs when the lesson started); with no
                # such body moving, the line in force is kept and the next gauge step reads again
                never = [bool(h) and not pe for h, pe in zip(has, self.pushed_ever or [False] * B)]
                f_new = hold_line_of(answer_p, never)
                if f_new is not None:
                    self.fatigue_line, self.fatigue_line_step = f_new, int(self._step)
                else:
                    last["fatigue_line_note"] = ("kept: no judged body the lesson never pushed moved in this window "
                                                 "(read again at the next gauge step)")
            last["fatigue_line"] = self.fatigue_line if self.fatigue_line is not None else f_line
            last["fatigue_line_step"] = self.fatigue_line_step
            if due:
                self._held_out_reading(X, P, u, encode_with, weights_now)
            if self.held_out is not None:
                last["held_out"] = dict(self.held_out)
        self.last = last
        self._due = bool(due)
        self._read_graph = torch.log(s_graph.clamp_min(RESPONSE_EPS)) if (due and s_graph.requires_grad) else None
        return lesson

    def _held_out_reading(self, X, P, u, encode_with, weights_now) -> None:
        """THE HELD-OUT READING (module docstring): at the first gauge step at least one window after the encoder was
        saved, the current window (all of it newer than that encoder) read through the saved encoder; then the encoder
        saved again. The first gauge step (the start) only saves it."""
        if encode_with is None or weights_now is None or self._step is None:
            return
        if self.snapshot is None or self.snapshot_step is None:
            self.snapshot, self.snapshot_step = weights_now(), int(self._step)
            return
        if int(self._step) - int(self.snapshot_step) < WINDOW * SAMPLE_EVERY:
            return
        B, n, F = X.shape
        enc_old = encode_with(self.snapshot)
        with torch.no_grad():
            z = enc_old(X.reshape(B * n, F)).reshape(B, n, -1).float()
            s_old = response(z)
            r_old = ratio_of(s_old, u)
            ref = (self.start or {}).get("ratio") or [float(v) for v in r_old]
            gain = [float(r) / float(r0) if r0 and r0 > 0 else 1.0 for r, r0 in zip(r_old, ref)]
            with torch.amp.autocast("cuda", enabled=False), torch.amp.autocast("cpu", enabled=False):
                cg = gain_matched_content(z, P, gain)
        self.held_out = dict(step=int(self._step), encoder_step=int(self.snapshot_step),
                             ratio=[float(v) for v in r_old], content_gain_matched=[float(v) for v in cg])
        self.snapshot, self.snapshot_step = weights_now(), int(self._step)

    def apply(self, lesson: torch.Tensor, params: Sequence[torch.Tensor], optimizer) -> dict:
        """Add the lesson's gradient to `params`' .grad AFTER production's clipping, each tensor's share rescaled to
        self.strength x that tensor's running gradient size (running_size). A tensor with no optimizer state yet
        (before its first update) is left out. On a gauge step it also reads the lesson's step per weight in learning
        rates and each body's first-order change of ln R_b from it (the attribution). Returns (and keeps) what was
        added."""
        params = [p for p in params if p.requires_grad]
        hyper = {}
        for group in optimizer.param_groups:
            b2 = float(group.get("betas", (0.9, 0.999))[1])
            for q in group["params"]:
                hyper[id(q)] = (b2, float(group.get("lr", 0.0)), float(group.get("eps", 1e-8)))
        rg, self._read_graph = self._read_graph, None
        due = bool(self._due and rg is not None)
        per_body = None
        if due:
            # each body's reading gradient (ln s_b on the real rows; ln u_b is data), before the lesson's own backward
            per_body = [torch.autograd.grad(rg[b], params, retain_graph=True, allow_unused=True)
                        for b in range(rg.shape[0])]
        grads = torch.autograd.grad(lesson, params, allow_unused=True)
        F = None if self.ring is None else int(self.ring.shape[-1])
        added, skipped, shares = 0, 0, []
        steps = {}
        for p, g in zip(params, grads):
            if g is None:
                continue
            b2, lr, eps = hyper.get(id(p), (0.999, 0.0, 1e-8))
            size = running_size(optimizer, p, b2)
            gr = _rms(g)
            if size is None or gr <= 0.0 or not math.isfinite(gr):
                skipped += 1
                continue
            target = self.strength * _rms(size)
            if target <= 0.0 or not math.isfinite(target):
                skipped += 1
                continue
            add = (g.detach() * (target / gr)).to(p.dtype)
            before = None if p.grad is None else _rms(p.grad)
            if p.grad is None:
                p.grad = add.clone()
            else:
                p.grad.add_(add)
            added += 1
            if before:
                shares.append(target / before)
            if due:
                # this weight's step from the lesson, in learning rates: Adam divides by this weight's own size
                steps[id(p)] = (p, add.detach().float() / (size.float() + eps), lr)
        self._applied_since += 1
        shares.sort()
        applied = dict(tensors=added, skipped=skipped, strength=self.strength,
                       share_of_clipped_gradient_median=(shares[len(shares) // 2] if shares else None))
        if due and steps:
            every = torch.cat([s.abs().flatten() for _p, s, _lr in steps.values()])
            applied.update(step_lr_rms=float(every.square().mean().sqrt()), step_lr_max=float(every.max()))
            first = next(((p, s) for p, s, _lr in steps.values() if p.dim() == 2 and F is not None
                          and p.shape[1] == F), None)
            if first is not None:
                fields = {}
                for name, idx in INPUT_FIELDS.items():
                    cols = [j for j in range(F) if j % N_BONE_FIELDS in idx]
                    sel = first[1][:, cols].abs()
                    fields[name] = dict(rms=float(sel.square().mean().sqrt()), max=float(sel.max()))
                applied["first_layer_step_lr"] = fields
            pace = []
            for gb in per_body:
                tot = 0.0
                for p, g_b in zip(params, gb):
                    if g_b is None or id(p) not in steps:
                        continue
                    _p, s, lr = steps[id(p)]
                    tot -= lr * float((g_b.detach().float() * s).sum())
                pace.append(tot)
            if self.attributed is None or len(self.attributed) != len(pace):
                self.attributed = [0.0] * len(pace)
            # the pace read now stands for every update since the last reading (at most GAUGE_EVERY)
            self.attributed = [a + v * self._applied_since for a, v in zip(self.attributed, pace)]
            self._applied_since = 0
            self.last["lesson_pace"] = pace
            self.last["attributed"] = list(self.attributed)
        self._due = False
        self.applied = applied
        return self.applied

    # ---------------------------------------------------------------- the record
    def gauges(self) -> dict:
        """The stats record's `proprio_token_repair` block (with the flag only). Lists are per body, index 0 = body
        1 as the dashboard numbers them."""
        g = dict(mode="within_body", version=VERSION, line=self.line, deaf=self.deaf, strength=self.strength,
                 skip=list(self.skip), window=WINDOW, sample_every=SAMPLE_EVERY,
                 min_samples=MIN_SAMPLES, gauge_every=GAUGE_EVERY, control_lambda=CONTROL_LAMBDA,
                 fatigue_step=FATIGUE_STEP, samples=int(self.count), restored_from=self.restored_from,
                 heading_free_rows=self.rows_transform is not None)

        def rnd(v):
            if isinstance(v, list):
                return [float("%.5g" % x) if isinstance(x, float) else x for x in v]
            return float("%.5g" % v) if isinstance(v, float) else v
        for k in ("lesson", "hold", "ratio", "ratio_posture_only", "spread", "spread_posture_only", "posture_spread",
                  "shortfall", "jitter", "jitter_share", "posture_jitter_share", "content", "content_control",
                  "content_gain_matched", "fatigue_answer", "fatigue_answer_posture_only", "fatigue_line",
                  "fatigue_line_step", "fatigue_line_note", "hold_share_since_gauge", "hold_any_share_since_gauge",
                  "hold_steps_since_gauge", "below_line", "pushed", "engaged", "pushed_ever", "held", "no_movement",
                  "lesson_pace", "attributed", "gauge_step", "ring_restarted", "start_retaken"):
            if k in self.last:
                g[k] = rnd(self.last[k])
        if self.start is not None:
            g["start"] = {k: rnd(v) for k, v in self.start.items()}
        if self.held_out is not None:
            g["held_out"] = {k: rnd(v) for k, v in self.held_out.items()}
        if self.applied:
            g["applied"] = {k: (float("%.5g" % v) if isinstance(v, float) else v) for k, v in self.applied.items()}
        g["scale"] = {"ratio": "each body's token movement per unit of its own posture movement over the 300-s window "
                               "(the token's and the posture's standard deviation over time, mean over numbers): 0 = "
                               "the token does not move with the posture; the line = the live bodies' level",
                      "ratio_posture_only": "the same with the velocity and capacity inputs held at their window "
                                            "mean: the part that answers the posture (the lesson pushes this one)",
                      "engaged": "the bodies the lesson pushes: engaged when read under DEAF, released at LINE",
                      "pushed_ever": "the bodies the lesson has pushed since the start (the hold acts on these only)",
                      "fatigue_line": "the hold's line in force: the median fatigue answer on the posture-only rows of "
                                      "the judged bodies the lesson never pushed, read at the start and again each "
                                      "window (300 s) (null: no live body at the start, the hold does not act)",
                      "fatigue_line_step": "the step the hold's line was last read",
                      "fatigue_answer_posture_only": "the fatigue answer on the posture-only rows: every pushed body's "
                                                     "at every step (the hold's reading), every moving body's on a "
                                                     "gauge step",
                      "hold_share_since_gauge": "per body, the share of the steps since the last gauge step in which "
                                                "the hold pulled it down",
                      "hold_any_share_since_gauge": "the share of the steps since the last gauge step in which the hold "
                                                    "acted on any body",
                      "held": "the pushed bodies whose posture-only fatigue answer read over the fatigue line (the "
                              "hold pulled them down this step)",
                      "hold": "1/2 x the sum of the held bodies' squared log excess over the fatigue line / bodies",
                      "shortfall": "ln line - ln ratio for a pushed body (0 at or above the line, or not engaged)",
                      "jitter_share": "the token's share of its own variance that changes between consecutive samples "
                                      "(0 = smooth, 1 = consecutive tokens unrelated)",
                      "posture_jitter_share": "the same share of the posture itself",
                      "content": "the version-4 posture content (a ten-block decoder's R2 at the penalty "
                                 "control_lambda): 0 = the token says nothing about the posture, 1 = exactly",
                      "content_gain_matched": "the same with the token's gain since the start divided out: a pure gain "
                                              "reads the start's content, movement without posture information less",
                      "content_control": "the same with the tokens shifted 150 s against the posture",
                      "fatigue_answer": "the token's mean change per number when every fatigue input moves by "
                                        "fatigue_step, over its response: 0 = the token ignores fatigue",
                      "held_out": "the window read through the encoder saved one window earlier (movement that encoder "
                                  "never trained on): its ratio and gain-matched content",
                      "lesson_pace": "each body's first-order change of ln ratio per update from the lesson's own step",
                      "attributed": "the lesson's summed first-order change of each body's ln ratio since the start",
                      "start": "the readings of the first full window, before the lesson's first update",
                      "spread": "the token's standard deviation over the 300-s window per number, mean over numbers: "
                                "0 = one token every moment",
                      "posture_spread": "the posture's standard deviation over the window per number, mean over "
                                        "numbers",
                      "below_line": "bodies pushed at the last step",
                      "no_movement": "bodies whose posture numbers did not change over the window (no ratio)",
                      "applied": "the lesson's gradient added after the clip, as a share of each encoder tensor's "
                                 "running gradient size; step_lr: the lesson's step per weight in learning rates"}
        return g

    def stamp(self) -> dict:
        """The receipt's `proprio_token_repair` entry."""
        return dict(version=VERSION, mode="within_body", line=self.line, deaf=self.deaf, strength=self.strength,
                    skip=list(self.skip), window=WINDOW, sample_every=SAMPLE_EVERY,
                    min_samples=MIN_SAMPLES, gauge_every=GAUGE_EVERY, control_lambda=CONTROL_LAMBDA,
                    fatigue_step=FATIGUE_STEP,
                    hold="each pushed body's fatigue answer held at the median posture-only answer of the judged "
                         "bodies never pushed, read at the start and again each window",
                    applied="after the gradient clip, scaled to each encoder tensor's running gradient size")

    # ---------------------------------------------------------------- runtime_state.pt
    def state_dict(self) -> dict:
        return dict(version=VERSION, count=int(self.count), ptr=int(self.ptr),
                    ring=None if self.ring is None else self.ring.detach().cpu().clone(),
                    start=None if self.start is None else dict(self.start),
                    attributed=None if self.attributed is None else list(self.attributed),
                    engaged=None if self.engaged is None else list(self.engaged),
                    pushed_ever=None if self.pushed_ever is None else list(self.pushed_ever),
                    snapshot=(None if self.snapshot is None else
                              {k: v.detach().cpu().clone() for k, v in self.snapshot.items()}),
                    snapshot_step=self.snapshot_step,
                    held_out=None if self.held_out is None else dict(self.held_out),
                    fatigue_line=self.fatigue_line, fatigue_line_step=self.fatigue_line_step,
                    applied_since=int(self._applied_since))

    def load_state_dict(self, sd: Optional[dict], device) -> str:
        """Restore the window (and this version's start, attributed lift, engaged and pushed flags, saved encoder and
        held-out reading); returns what happened (for the load's log line). Every version from 3 on keeps the same raw
        rows; a window saved by versions 1 and 2 holds 200 samples (50 s), another shape, and starts empty. What another
        version saved beside the window is not restored: version 4's start read an R2, version 5 had no hinge; the start
        is read again on the first full window."""
        self.reset()
        if not sd or sd.get("version") not in READS_VERSIONS or not torch.is_tensor(sd.get("ring")):
            self.restored_from = None
            return "starts empty (none saved)"
        mine = sd.get("version") == VERSION
        if mine and isinstance(sd.get("start"), dict):
            self.start = dict(sd["start"])
        if mine and isinstance(sd.get("attributed"), list):
            self.attributed = [float(v) for v in sd["attributed"]]
        if mine and isinstance(sd.get("engaged"), list):
            self.engaged = [bool(v) for v in sd["engaged"]]
        if mine and isinstance(sd.get("pushed_ever"), list):
            self.pushed_ever = [bool(v) for v in sd["pushed_ever"]]
        if mine and isinstance(sd.get("snapshot"), dict):
            self.snapshot = {k: v.to(device) for k, v in sd["snapshot"].items()}
            self.snapshot_step = sd.get("snapshot_step")
        if mine and isinstance(sd.get("held_out"), dict):
            self.held_out = dict(sd["held_out"])
        if mine and "fatigue_line" in sd:
            fl = sd.get("fatigue_line")
            self.fatigue_line = float(fl) if isinstance(fl, (int, float)) else None
            self.fatigue_line_step = sd.get("fatigue_line_step")
        self._applied_since = int(sd.get("applied_since") or 0) if mine else 0
        ring = sd["ring"]
        if ring.dim() != 3 or ring.shape[1] != WINDOW:
            self.restored_from = None
            return "starts empty (the saved window has another shape %s)" % (tuple(ring.shape),)
        self.ring = ring.to(device=device, dtype=torch.float32).clone()
        self.count = max(0, min(int(sd.get("count", 0)), WINDOW))
        self.ptr = int(sd.get("ptr", 0)) % WINDOW
        self.restored_from = "runtime_state.pt"
        return "restored (%d samples of %d bodies%s)" % (
            self.count, ring.shape[0], "; the start, the attributed lift, the engaged and pushed flags" if self.start
            else
            ("" if mine else "; the start of %s is not this version's reading: read again" % sd.get("version")))
