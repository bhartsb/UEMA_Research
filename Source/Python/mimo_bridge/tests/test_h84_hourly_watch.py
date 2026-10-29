# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Exercise the H84 watcher (docs/h84_launch_2026-09-18/hourly_watch.py) without an hour or a trainer.

It differs from the H67-H83 watcher in judging erosion against the run's OWN first
hourly window instead of a trained lineage's levels (H84 starts from fresh weights), in
two stop rules declared at launch (replay loss > 3x hour 1; steps per second under one
third of hour 1), in judging token variance from run-hour 3 only, and in alerting on
failed or stale checkpoints.

Since 2026-09-23 every printed line obeys DESIGN.md section 11 (no bare score decimals where a
person reads): the tests below that pinned "0.12", "cosine 0.237" or "0.0 vs line 0.0001" were
rewritten to the counts, units and percentages the lines now print, and two tests scan the whole
output - on a real segment 31 record and on a record built from the producers' own summary
functions - with tools/check_no_bare_decimals.bare_decimals.
"""
import collections
import json
import math
import random
from pathlib import Path
import runpy
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from mimo_bridge.tools.check_no_bare_decimals import bare_decimals, unanchored_numbers


WATCHER = Path(__file__).resolve().parents[4] / 'docs/h84_launch_2026-09-18/hourly_watch.py'
REAL_SAVED = Path(__file__).resolve().parents[4] / 'Saved' / 'mimo_bridge'
SEG31 = REAL_SAVED / 'experiments' / 'H84_fresh_drivers_20260918_seg31_toy_and_push'
HEALTHY_DISK = collections.namedtuple('usage', 'total used free')(1e12, 0.0, 500e9)


def healthy_resources(monkeypatch):
    monkeypatch.setattr(shutil, 'disk_usage', lambda _p: HEALTHY_DISK)
    # the machine lines read the commit charge and the card live; hold them healthy and fixed so
    # a test never depends on this machine (and never waits on nvidia-smi)
    from mimo_bridge import machine_stats
    from mimo_bridge.tools import commit_guard
    monkeypatch.setattr(machine_stats, 'nvidia_smi_memory', lambda: (5000.0, 8188.0))
    monkeypatch.setattr(commit_guard, 'commit_reading', lambda: (40.0, 68.8, 20.0))
    try:
        import psutil
    except ImportError:
        return
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: collections.namedtuple('vm', 'available')(64e9))


def row(**changes):
    r = dict(type='mimo', step=500, timestamp=10000, elapsed_s=7200, nan_steps=0,
             model_steps_per_s=0.6, replay_loss=0.5, checkpoint_failures=0, peak_vram_mb=5200,
             token_temporal_std=dict(vision=.6, tactile=.6),
             causal_pe_per_modality=dict(vision=.01, tactile=.04, proprioception=.005),
             bout_body_speed=.00154, follow_bout=.1, task_hand_fov=.001,
             consequence=dict(rate=0.3, goal_rate=0.2, kinds_mean=0.4, weight=0.5),
             intention=dict(gap_closed_pct=None, closer_than_start=False),
             cmd_response_cos=0.1, cmd_response_cos_held=0.12, task_mobile_hand=0.0,
             hself=dict(slots_filled=6400, slots_live=6400, occ_entropy=0.98, l2_sig=-1, l3_sig=-1, l2_ticks=7,
                        l2_ticks_needed=25, l2_steps_to_first_evidence=180000, l3_ticks=0, l3_ticks_needed=25,
                        l3_steps_to_first_evidence=2500000))
    r.update(changes)
    return r


def watch(tmp_path, monkeypatch, baseline=None, criteria=None, checkpoint=True, args='{}', **changes):
    healthy_resources(monkeypatch)
    # the storage lines read the archiver's status and the D: manifests: never the real ones here
    monkeypatch.setenv('UEMA_WATCH_SAVED', str(tmp_path / '_saved'))
    monkeypatch.setenv('UEMA_WATCH_ARCHIVE', str(tmp_path / '_archive'))
    for stale in ('control.json', 'stop_reason.json', 'watcher_baselines.json', 'declared_criteria.json'):
        if (tmp_path / stale).exists():
            (tmp_path / stale).unlink()           # watch() is called twice in some tests
    (tmp_path / 'args.json').write_text(args)
    (tmp_path / 'replay_training_stats.jsonl').write_text(json.dumps(row(**changes)) + '\n')
    if baseline is not None:
        (tmp_path / 'watcher_baselines.json').write_text(json.dumps(baseline))
    if criteria is not None:
        (tmp_path / 'declared_criteria.json').write_text(json.dumps(criteria))
    ck = tmp_path / 'checkpoints' / 'ckpt_step_400'
    if checkpoint:
        ck.mkdir(parents=True, exist_ok=True)
        (ck / 'meta.json').write_text('{"step": 400}')
    elif ck.exists():
        shutil.rmtree(tmp_path / 'checkpoints')
    monkeypatch.setattr(sys, 'argv', [str(WATCHER), str(tmp_path), '3'])
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    monkeypatch.setattr(time, 'time', lambda: 10000)
    runpy.run_path(str(WATCHER), run_name='__main__')
    control = tmp_path / 'control.json'
    return json.loads(control.read_text()) if control.exists() else None


BASE = dict(vision=.01, tactile=.04, replay_loss=0.5, steps_per_s=0.6, peak_vram_mb=5200, measured_at_step=1)


def test_the_first_check_measures_the_baselines_and_a_healthy_run_is_not_stopped(tmp_path, monkeypatch, capsys):
    assert watch(tmp_path, monkeypatch) is None
    base = json.loads((tmp_path / 'watcher_baselines.json').read_text())
    assert base['vision'] == pytest.approx(.01) and base['replay_loss'] == pytest.approx(0.5)
    assert base['steps_per_s'] == pytest.approx(0.6) and base['measured_at_step'] == 500
    out = capsys.readouterr().out
    assert 'OWN BASELINES measured' in out
    assert 'CHECKPOINT ALERT' not in out and not (tmp_path / 'stop_reason.json').exists()


@pytest.mark.parametrize('changes,reason', [
    (dict(replay_loss=1.6), 'replay loss'),                                   # > 3 x 0.5
    (dict(model_steps_per_s=0.19), 'steps per second'),                       # < 0.6 / 3
    (dict(causal_pe_per_modality=dict(vision=.021, tactile=.04)), 'vision causal'),
    (dict(causal_pe_per_modality=dict(vision=.01, tactile=.13)), 'tactile causal'),
])
def test_own_baseline_stop_rules_fire_after_two_checks(tmp_path, monkeypatch, changes, reason):
    assert watch(tmp_path, monkeypatch, baseline=BASE, **changes)['cmd'] == 'snapshot_and_stop'
    receipt = json.loads((tmp_path / 'stop_reason.json').read_text())
    assert receipt['resume_policy'] == 'repair_required' and reason in receipt['reason']
    assert 'own hour-1 baseline' in receipt['reason']
    assert not bare_decimals(receipt['reason']), receipt['reason']


@pytest.mark.parametrize('changes', [
    dict(replay_loss=1.4),                 # 2.8x: under the 3x line
    dict(model_steps_per_s=0.21),          # above one third
])
def test_own_baseline_rules_hold_under_their_lines(tmp_path, monkeypatch, changes):
    assert watch(tmp_path, monkeypatch, baseline=BASE, **changes) is None


def test_the_baseline_window_itself_is_never_judged(tmp_path, monkeypatch):
    """Check 1 measures; with no earlier baseline, the same window cannot breach its own line."""
    assert watch(tmp_path, monkeypatch, replay_loss=9.0) is None


def test_token_variance_is_judged_from_run_hour_three(tmp_path, monkeypatch):
    low = dict(token_temporal_std=dict(vision=.01, tactile=.6))
    assert watch(tmp_path, monkeypatch, baseline=BASE, elapsed_s=7200, **low) is None
    assert watch(tmp_path, monkeypatch, baseline=BASE, elapsed_s=3 * 3600, **low)['cmd'] == 'snapshot_and_stop'
    reason = json.loads((tmp_path / 'stop_reason.json').read_text())['reason']
    assert 'vision variance' in reason and '6.7% / tactile 400% of the stop line' in reason


def test_failed_or_stale_checkpoints_alert_but_do_not_stop(tmp_path, monkeypatch, capsys):
    assert watch(tmp_path, monkeypatch, baseline=BASE, checkpoint_failures=2,
                 checkpoint_last_error='OSError: The device is not ready') is None
    out = capsys.readouterr().out
    assert 'CHECKPOINT ALERT: saves are failing' in out and 'device is not ready' in out
    assert watch(tmp_path, monkeypatch, baseline=BASE, checkpoint=False) is None
    assert 'no complete checkpoint in the last 2 h' in capsys.readouterr().out


def test_memory_growth_alerts_but_does_not_stop(tmp_path, monkeypatch, capsys):
    assert watch(tmp_path, monkeypatch, baseline=BASE, peak_vram_mb=5800) is None
    assert 'MEMORY ALERT: peak allocation 5800 MiB is 600 above own hour-1 5200' in capsys.readouterr().out
    assert watch(tmp_path, monkeypatch, baseline=BASE, peak_vram_mb=5600) is None
    assert 'MEMORY ALERT' not in capsys.readouterr().out


def test_declared_criteria_alert_and_stop_as_declared(tmp_path, monkeypatch, capsys):
    crit = [dict(name='hand_strikes', path='task_mobile_hand', op='<', line=0.0001, from_run_hour=0, mode='alert', note='n'),
            dict(name='waits', path='task_mobile_hand', op='<', line=0.0001, from_run_hour=9, mode='stop', note='n')]
    assert watch(tmp_path, monkeypatch, baseline=BASE, criteria=crit) is None
    out = capsys.readouterr().out
    assert ("criterion 'hand_strikes': task_mobile_hand reads 0% of its line (a reading under the line "
            "is a breach) -> BREACHED") in out
    assert "criterion 'waits': waits until run-hour 9" in out
    crit[1]['from_run_hour'] = 0
    assert watch(tmp_path, monkeypatch, baseline=BASE, criteria=crit)['cmd'] == 'snapshot_and_stop'
    assert "declared criterion 'waits'" in json.loads((tmp_path / 'stop_reason.json').read_text())['reason']


def test_non_finite_and_low_disk_still_stop(tmp_path, monkeypatch):
    assert watch(tmp_path, monkeypatch, baseline=BASE, nan_steps=1)['cmd'] == 'snapshot_and_stop'
    assert 'non-finite' in json.loads((tmp_path / 'stop_reason.json').read_text())['reason']
    monkeypatch.setattr(shutil, 'disk_usage', lambda _p: HEALTHY_DISK._replace(free=1e9))
    (tmp_path / 'args.json').write_text('{}')
    (tmp_path / 'replay_training_stats.jsonl').write_text(json.dumps(row()) + '\n')
    (tmp_path / 'control.json').unlink()
    monkeypatch.setattr(sys, 'argv', [str(WATCHER), str(tmp_path), '3'])
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    monkeypatch.setattr(time, 'time', lambda: 10000)
    runpy.run_path(str(WATCHER), run_name='__main__')
    assert json.loads((tmp_path / 'control.json').read_text())['cmd'] == 'snapshot_and_stop'
    assert 'free disk' in json.loads((tmp_path / 'stop_reason.json').read_text())['reason']


def test_the_execution_line_names_the_250_ms_gauge_without_printing_a_cosine(tmp_path, monkeypatch, capsys):
    """2026-09-19: the execution line leads with the 250 ms gauge. 2026-09-23 (DESIGN section 11):
    that gauge is a direction match (a cosine), which is never printed; the line says so, and the
    stop rule on the bout reading still judges it."""
    assert watch(tmp_path, monkeypatch, baseline=BASE) is None
    out = capsys.readouterr().out
    assert "how closely the body's 250 ms turn follows its command is not yet measured as a count" in out
    assert 'the babble-bout reading is above its stop line' in out
    assert watch(tmp_path, monkeypatch, baseline=BASE, follow_bout=-0.05)['cmd'] == 'snapshot_and_stop'
    reason = json.loads((tmp_path / 'stop_reason.json').read_text())['reason']
    assert 'babble-bout direction match' in reason and not bare_decimals(reason)


def test_the_hself_line_prints_the_keys_the_record_carries(tmp_path, monkeypatch, capsys):
    """2026-09-19: the line read a key the record never had and printed None for 19 hours."""
    assert watch(tmp_path, monkeypatch, baseline=BASE) is None
    out = capsys.readouterr().out
    assert "windows lived 7 of the 25 needed, 180000 steps to go" in out
    assert "hours level: -1, 0 of 25, 2500000 steps to go" in out
    assert "model steps that belong to a slot" not in out, "a record without the new gauges must not print a line of Nones"
    assert watch(tmp_path, monkeypatch, baseline=BASE, hself=None) is None
    out = capsys.readouterr().out
    assert "hself(dark): unmeasured - the newest record carries no hself block" in out and "None/" not in out


def test_the_watcher_says_whether_the_table_is_about_todays_moments(tmp_path, monkeypatch, capsys):
    """2026-09-19, audit P238: the share of moments that belong to a slot and the count of slots that are
    copies say whether the evidence counts are about today's moments at all. The nearness of a moment to
    its slot is a cosine, which DESIGN section 11 does not print."""
    hs = dict(slots_filled=6400, slots_live=6400, occ_entropy=0.98, l2_sig=-1, l3_sig=-1, l2_ticks=7,
              l2_ticks_needed=25, l2_steps_to_first_evidence=180000, l3_ticks=0, l3_ticks_needed=25,
              l3_steps_to_first_evidence=2500000, moments_with_a_slot=0.0, best_slot_cos=0.2367,
              slots_twinned=0, nearest_other_cos_median=0.958)
    assert watch(tmp_path, monkeypatch, baseline=BASE, hself=hs) is None
    out = capsys.readouterr().out
    assert "model steps that belong to a slot 0 of every 100 model steps (0 = none" in out
    assert "how alike a model step and its nearest slot are: not yet measured as a count" in out
    assert "slots that are copies of another slot 0 of 6400" in out
    # H84 segments 6 and 7 wrote the share under its first name; the line reads that too
    first = dict(hs, moments_with_a_row=0.25)
    first.pop("moments_with_a_slot")
    assert watch(tmp_path, monkeypatch, baseline=BASE, hself=first) is None
    assert "model steps that belong to a slot 25 of every 100 model steps" in capsys.readouterr().out


def test_the_watcher_reports_the_named_things_and_the_reader(tmp_path, monkeypatch, capsys):
    """2026-09-20: the self-model's slow levels now also watch NAMED things, and a reader can
    turn a proven link into a goal the planner aims at. The hourly check must say how many
    things are watched, which links are strongest, how far the evidence has to go, and whether
    the reader's gate is open - the one line that says whether behaviour is being changed."""
    es = {"names": ["caregiver", "toy_ball", "mobile"], "teacher": "wellness",
          "presence_mean": [0.8, 0.3, 0.9], "l2_link": [0.42, -0.05, 0.11],
          "l2_entity_windows": [7, 7, 7], "l2_ticks": 7, "windows_needed": 11,
          "l2_counted_names": [], "l2_window_moments": 10000,
          "reader": {"gate_open": False, "applied_weight": 0.0, "weight": 0.25,
                     "why": "no entity has passed the evidence test yet, so the reader has nothing to read"}}
    assert watch(tmp_path, monkeypatch, baseline=BASE, entity_self=es) is None
    out = capsys.readouterr().out
    assert "entity self (dark): 3 named things, learning from wellness" in out
    assert "7 windows lived of the 11 a link needs" in out and "counted: none yet" in out
    assert "caregiver present on 80 of every 100 model steps, 7 windows of its own" in out
    assert out.index("caregiver present") < out.index("mobile present"), "the strongest link comes first"
    assert "the reader: gate shut, so nothing is added" in out and "nothing to read" in out
    # and when the gate opens, the line says so in those words
    es2 = dict(es, reader=dict(es["reader"], gate_open=True, applied_weight=0.25,
                               why="the real reader tracks wellness better than the shuffled control"))
    (tmp_path / "open").mkdir()
    assert watch(tmp_path / "open", monkeypatch, baseline=BASE, entity_self=es2) is None
    out = capsys.readouterr().out
    assert "gate OPEN" in out and "weight applied 100% of the weight it asks for" in out
    # a run without the entity levels prints no line of empties
    (tmp_path / "off").mkdir()
    assert watch(tmp_path / "off", monkeypatch, baseline=BASE) is None
    assert "entity self" not in capsys.readouterr().out


# ------------------------------------------------------------ gain-of-function, storage and machine
def _module():
    """The watcher imported for its line builders; nothing runs on import."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('h84_hourly_watch_lines', WATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Blocks whose producer needs a full LemLite to summarise (efference_window_summary,
# synergy_window_summary, the wellness_rank wrapper's mode/weights) are typed from the producers'
# docstrings; every other block below is BUILT by the producer's own summary function
# (review ops-storage finding 5: a fixture typed from DESIGN could never fail on a renamed field).
EFFERENCE = dict(body_cm_err_next=0.5, body_cm_do_nothing_next=1.0, body_cm_shuffled_next=0.9,
                 body_cm_err_latch=2.0, body_cm_do_nothing_latch=4.0, body_cm_shuffled_latch=3.5,
                 body_deg_err_latch=3.0, body_deg_do_nothing_latch=6.0, body_deg_shuffled_latch=5.0,
                 gaze_deg_err=[1.0, 2.0, 3.0], gaze_deg_do_nothing=[2.0, 4.0, 6.0],
                 gaze_deg_shuffled=[1.9, 3.9, 5.9], loss_body=0.01, loss_gaze=0.02,
                 liveness={"body.0": 0.62, "gaze.0": 0.31})
SYNERGY = dict(dim=32, orders=900, explained_share=0.55, explained_share_unweighted=0.5,
               effective_dim=4.2, z_rms=0.96, following_mean=0.32, movement_mean=1.0,
               babble_sigma=0.3, babble_turn_rad=0.8, babble_target_rad=0.798, babble_move_cm=1.7,
               babble_at_rail=0.02, cmd_mag_rad=0.5, lesson_loss=0.1, residual_norm=0.02,
               liveness={"net.0": 1.0, "net.2": 0.97})


def contact_block(contract=False):
    """contact_control from ContactTracker.summary over 160 moments of 2 bodies; contract=True adds
    the fields the model lane is adding (onset AUCs, two gates)."""
    from lem.model import contact_head as CH
    rng = np.random.default_rng(3)
    tr = CH.ContactTracker()
    P, B = CH.N_PATCHES, 2
    habit = [{} for _ in range(B)]
    prev = rng.random((B, P, 3)) < 0.1
    for step in range(160):
        cur = rng.random((B, P, 3)) < 0.1
        q = np.clip(rng.random((B, P, 3)) * 0.4 + cur * 0.4, 0, 1)
        onsets = [[dict(pair=[CH.PATCH_NAMES[1], CH.PATCH_NAMES[5]], structural=False)] if step % 3 == 0
                  else [] for _ in range(B)]
        # the control's order foresees exactly what the own order does: no advantage, gate shut
        qc = q.copy() if step % 7 == 0 else None
        tr.score(step, q, prev, cur, np.ones(B, bool), onsets, habit, world_report=[1, 0], q_control=qc)
        prev = cur
    out = tr.summary(liveness={"trunk.0": 0.8, "patch.0": 0.42})
    out["control_period"], out["channel"] = 7, {"weight": 0.5, "in_ranking": True}
    if contract:
        out.update(onset_auc_self=0.737, onset_auc_world=0.81, control_onset_auc_self=0.824,
                   control_onset_auc_world=0.79, gate_self_open=False, gate_world_open=True,
                   pool_spread={"patch_pool": 0.091})
    return out


def controllability_block(contract=False):
    from lem.model import wellness_channels as WC
    tr = WC.ControllabilityTracker(n_handles=26, gaze_idx=23, hold=5, floors=dict(WC.CONTROL_FLOOR))
    tr.win.update(orders=80, measured=20, agree=6.0, control=0.4, control_n=20, command_agree=5.0,
                  command_n=20, source="efference", by_author={"planner": [4.0, 10], "reflex_noise": [2.0, 10]})
    for i in range(WC.CONTROL_GATE_READINGS):
        tr.gate_readings.append(0.05 if i % 2 else -0.05)       # no advantage: the gate stays shut
    out = tr.summary(B=2)
    if contract:
        out["difference_mean"] = round(out["agreement_mean"] - out["control_mean"], 4)
    return out


def learning_progress_block(contract=False):
    import torch
    from lem.model import wellness_channels as WC
    lp = WC.LearningProgressTracker(fast=5, slow=50)
    for t in range(60):
        lp.update("vision", torch.tensor([1.0 - 0.01 * t, 1.0]), torch.tensor([1.2, 1.2]))
        lp.update("proprioception", torch.tensor([0.5, 0.5]), torch.tensor([0.4, 0.4]))
        lp.tick_window(2)
    out = lp.summary(2)
    if contract:
        out["control_mean"] = -0.03
    return out


def rank_block(contract=False):
    import torch
    from lem.model import wellness_channels as WC
    rs = WC.RankShares()
    names = ["integrity", "controllability", "curiosity"]
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        contrib = torch.randn(12, len(names), 2, generator=g)
        ranked = contrib.sum(1)
        gated = torch.zeros(12, 2, dtype=torch.bool)
        gated[0, 0] = True
        rs.add(contrib, ranked.masked_fill(gated, float("-inf")), gated, torch.zeros(2, dtype=torch.bool), names)
    out = dict(mode="normalized", curiosity_mode="additive",
               weights=dict(integrity=2.0, controllability=1.0, curiosity=0.25),
               weights_in_force=dict(integrity=2.0, controllability=0.0, curiosity=0.25))
    out.update(rs.summary())
    if contract:
        out["goal_age"] = dict(mean_share_back=0.62, oldest_tenth_share=0.31, newest_tenth_share=0.04, events=40)
        out["weight_share_by_channel"] = dict(integrity=2.0 / 3.25, controllability=1.0 / 3.25, curiosity=0.25 / 3.25)
    return out


def causal_chain_block(contract=False):
    from mimo_bridge.causal_chain import CausalChainRecorder
    rec = CausalChainRecorder(None, hold=5)
    w = rec.win
    w.events, w.planner_events, w.latch_predictions = 40, 10, 30
    rec.tie_band = 0.01
    w.outcome, w.rank, w.alternatives_n = ['best'] * 6 + ['behind'] * 4, [1] * 6 + [3] * 4, 4
    w.reached, w.exec_cm = [1.0] * 9 + [0.0], [3.5] * 9
    import torch
    t = torch.tensor
    w.add_err("pred_latch", {"vision": t([0.8]), "tactile": t([0.5])})
    w.add_err("persist_latch", {"vision": t([1.0]), "tactile": t([0.4])})
    w.add_err("pred_next", {"vision": t([0.9])})
    w.add_err("persist_next", {"vision": t([1.0])})
    w.wchg["pred_latch"], w.wchg["real_latch"] = [0.1, 0.2, 0.3, 0.1], [0.0, 0.1, 0.4, 0.2]
    out = rec.window_summary()
    if contract:
        out.update(chosen_best_share=0.1, chosen_tied_share=0.5, tie_band=0.01,
                   gap_to_best_alt_mean=0.002, candidate_spread_mean=0.004, plan_exec_distance_cm=3.2)
        out.pop("planner_share_of_executed", None)
    return out


def vestibular_block():
    from mimo_bridge.vestibular_sense import VestibularWindow
    from mimo_bridge.vestibular_adapter import VESTIBULAR_DIM, VESTIBULAR_FIELDS
    win = VestibularWindow()
    rng = np.random.default_rng(1)
    ix = {n: i for i, n in enumerate(VESTIBULAR_FIELDS)}
    for step in range(12):
        frames = []
        for b in range(2):
            v = rng.normal(0, 0.2, VESTIBULAR_DIM).astype(np.float32)
            v[ix["head_acc_z"]] = 1.0
            v[ix["trunk_up_z"]] = 1.0
            v[ix["valid"]] = 1.0
            v[ix["support_trunk"]] = 0.8
            v[ix["limit_load"]] = 0.05
            frames.append(SimpleNamespace(vestibular=v))
        model = SimpleNamespace(_last_source_mask={"vestibular": "stale" if step == 0 else "real"},
                                last_prediction_audit={"vestibular": dict(transitions=2, prediction_mse=0.03,
                                                                          persistence_mse=0.04)})
        win.add(frames, model)
    return win.summary()


def gof_record(contract=False):
    return dict(type='mimo', causal_chain=causal_chain_block(contract), efference=EFFERENCE,
                controllability=controllability_block(contract),
                learning_progress=learning_progress_block(contract),
                wellness_rank=rank_block(contract), contact_control=contact_block(contract),
                synergy=SYNERGY, vestibular=vestibular_block(),
                next_frame_norm=dict(weight_effective=dict(vision=1.1, audio=0.9)))


def test_every_gain_of_function_block_prints_in_plain_words_or_says_unmeasured():
    m = _module()
    absent = m.gof_lines([dict(type='mimo')])
    assert len(absent) == 9 and all('unmeasured' in line for line in absent)
    rec = gof_record()
    text = '\n'.join(m.gof_lines([rec, rec]))
    assert 'LINE FAILED' not in text, text
    assert '60 commands predicted before they were sent' in text          # summed over the hour
    assert 'in 60 of every 100 planning events (field chosen_best_share)' in text
    assert 'a random pick scores 20 of every 100' in text
    assert 'vision +20.0%, tactile -25.0%' in text                          # 1 - pred/persist
    assert 'sat 3.5 cm from the plan' in text
    assert 'closes 50.0% of the gap' in text and '1.00/2.00/3.00 degrees at 50/100/150 ms' in text
    assert 'measured on 25 of every 100 commands' in text                  # a share of COMMANDS
    assert 'agreed +30% of the way' in text
    assert "with ANOTHER body's command from this body's state (the shuffled-command control, field control_mean) +2.0%" in text
    assert 'evidence gate shut - over its last readings' in text
    assert 'value into the goal choice 0 while the gate is shut' in text
    assert 'frozen weights, which cannot learn, read -7.1% to -1.6%' in text
    assert 'goal age not in this record' in text
    assert 'computed from weights_in_force' in text
    assert 'habituation: the average pay factor' in text and '(100% = a new pair, lower = repeated)' in text
    assert 'success_paired' in text and 'against' in text
    assert 'control_onset_auc_self' in text and '0.5 = a coin flip' not in text
    assert 'evidence gates: self shut, world shut' in text
    assert 'paid into wellness 0 while the gate is shut' in text
    assert 'item 5' in text and 'not item 5' in text and '28.6 degrees' in text
    assert 'the sense was a real reading on 22 of 24 model steps' in text  # real_steps against steps
    assert 'head tilt from lying on its back' in text and 'neck tilt' not in text
    assert 'within 2.0 degrees of a joint limit' in text and ' rad' not in text
    assert 'trunk pitch' in text and '(0 = horizontal, 90 = head straight up)' in text
    assert 'vision 110%, audio 90%' in text
    assert not bare_decimals(text), bare_decimals(text)
    assert not unanchored_numbers(text), unanchored_numbers(text)


def test_the_contract_fields_are_read_when_present_and_named():
    """The 2026-09-23 field contract between the lanes: each new field is read when the record carries
    it, and the line names the field it used."""
    m = _module()
    rec = gof_record(contract=True)
    text = '\n'.join(m.gof_lines([rec]))
    assert 'LINE FAILED' not in text, text
    assert ('ranks a real own-body touch onset above a quiet patch in 74 of every 100 pairs (onset_auc_self) '
            'against 82 of every 100 pairs for the base-rate control') in text
    assert 'evidence gates: self shut, world OPEN (gate_self_open, gate_world_open)' in text
    assert 'paid into wellness' in text and 'points per 1,000 model steps of one body' in text
    assert 'own command minus control' in text and '(difference_mean)' in text
    assert "from this body's state (the shuffled-command control, field control_mean) +2.0%" in text
    assert 'same-model-step control -3.0% (field control_mean)' in text
    assert '62% of the way from the newest entry of the goal ring (0%) to the oldest (100%)' in text
    assert '31 of every 100 goals came from the oldest tenth' in text
    assert "share of the weights (weight_share_by_channel)" in text
    assert 'and tied with the best of them in 50 of every 100 (chosen_tied_share)' in text
    assert 'spread 0.4 times the tie band' in text
    assert 'sat 3.2 cm from the plan' in text and 'plan_exec_distance_cm' in text
    assert not bare_decimals(text), bare_decimals(text)
    assert not unanchored_numbers(text), unanchored_numbers(text)


def test_the_contact_line_compares_the_same_onsets_and_beats_the_base_rate_not_a_coin_flip():
    """review b4-self-contact M4 and ops-storage finding 5: success_self counts every onset and
    success_control only the control's moments; the same-onset pair is success_paired against
    success_control. The head must beat the base-rate control, not 0.5."""
    m = _module()
    cc = dict(onsets_self=3, onsets_world=5, success_self=0.21, success_paired=0.53, success_control=0.47,
              success_world=0.2, head_auc=0.89, control_auc_bits=dict(any=0.95, self=0.9, world=0.93),
              habituation_mean=0.25, paid_mean=0.0, gate_open=False)
    text = '\n'.join(m.gof_lines([dict(type='mimo', contact_control=cc)]))
    assert ('on the SAME onsets: 53 of every 100 self-touch onsets with this body\'s own command (success_paired) '
            'against 47 of every 100 with another body\'s command (success_control)') in text
    assert '89 of every 100 pairs (head_auc) against 95 of every 100 pairs for the base-rate control' in text
    assert 'the average pay factor 25% (100% = a new pair, lower = repeated)' in text
    assert '0 while the gate is shut' in text
    # without success_paired the line says the two are not the same onsets
    cc.pop('success_paired')
    text = '\n'.join(m.gof_lines([dict(type='mimo', contact_control=cc)]))
    assert 'not the same onsets' in text and 'success_paired is not in this record' in text


def _world_onsets_block():
    """contact_control from ContactTracker.summary: a world touch begins on one patch every other
    moment; the head foresees it only on the moments the control runs (every 7th), so the head's
    rate over EVERY onset (success_world) differs from its rate on the control's onsets
    (success_paired_world)."""
    from lem.model import contact_head as CH
    tr = CH.ContactTracker()
    P, B = CH.N_PATCHES, 2
    prev = np.zeros((B, P, 3), bool)
    for step in range(700):
        cur = np.zeros((B, P, 3), bool)
        p = step % P
        if step % 2 == 0:
            cur[:, p, 2] = cur[:, p, 0] = True
        q = np.full((B, P, 3), 0.01)
        if step % 7 == 0:
            q[:, p, 2] = q[:, p, 0] = 0.99
        qc = np.full((B, P, 3), 0.01) if step % 7 == 0 else None
        tr.score(step, q, prev, cur, np.ones(B, bool), [[] for _ in range(B)], [{} for _ in range(B)],
                 world_report=[1, 1], q_control=qc)
        prev = cur
    return tr.summary()


def test_outside_touches_are_compared_on_the_same_onsets():
    """verify-fix-ops 2026-09-23: b4 review M4's mismatch on the world half of the line. The line put
    success_world (every onset: 14 of 100 here) beside success_control_world (only the control's
    moments: 0 of 100); on the same onsets the own order foresaw 100 of 100 (success_paired_world)."""
    m = _module()
    cc = _world_onsets_block()
    assert cc['success_paired_world'] == 1.0 and 0 <= cc['success_control_world'] < .2
    assert cc['success_world'] < cc['success_paired_world']
    text = '\n'.join(m.gof_lines([dict(type='mimo', contact_control=cc)]))
    assert ("outside touches on the SAME onsets: 100 of every 100 with this body's own command "
            f"(success_paired_world) against {m.of100(cc['success_control_world'])} with another body's command "
            "(success_control_world)") in text
    assert 'outside touches 14 of every 100 against' not in text
    # a record without the paired field says the two are not the same onsets
    cc.pop('success_paired_world')
    text = '\n'.join(m.gof_lines([dict(type='mimo', contact_control=cc)]))
    assert 'not the same onsets; success_paired_world is not in this record' in text
    assert not bare_decimals(text), bare_decimals(text)


def test_the_gate_line_is_named_from_the_record_never_as_fixed_text():
    """verify-fix-ops 2026-09-23: the line said "twice the readings' own standard error" for every
    gate; the model lane's controllability gate opens above 3 and, once open, closes below 1 of those
    errors (gate_lines_noise). A record without gate_lines_noise gets no multiplier at all."""
    m = _module()
    base = dict(orders=80, measured_share=0.25, agreement_mean=0.3, control_mean=0.02,
                difference_mean=0.28, control_kind='shuffled order', gate_mean=0.21, gate_line=0.0015,
                gate_lines_noise={'open': 3.0, 'close': 1.0})
    text = '\n'.join(m.gof_lines([dict(type='mimo', controllability=dict(base, gate_open=True))]))
    assert "its closing line, 1 time the readings' own standard error" in text and 'twice' not in text
    assert 'evidence gate OPEN' in text
    text = '\n'.join(m.gof_lines([dict(type='mimo', controllability=dict(base, gate_open=False))]))
    assert "its opening line, 3 times the readings' own standard error" in text
    assert 'value into the goal choice 0 while the gate is shut' in text
    v1 = controllability_block()
    v1.pop('gate_lines_noise', None)                   # a historical record predating hysteresis
    text = '\n'.join(m.gof_lines([dict(type='mimo', controllability=v1)]))
    assert "the line the producer drew from the readings' own standard error (field gate_line)" in text
    assert 'twice' not in text
    assert not bare_decimals(text), bare_decimals(text)


def test_a_contact_gate_whose_state_is_not_recorded_is_not_called_shut():
    m = _module()
    cc = dict(onsets_self=3, onsets_world=5, success_paired=0.5, success_control=0.4, paid_mean=0.02,
              habituation_mean=0.5)
    text = '\n'.join(m.gof_lines([dict(type='mimo', contact_control=cc)]))
    assert 'evidence gate state unmeasured' in text
    assert 'paid into wellness 20 points per 1,000 model steps of one body' in text
    assert '0 while the gate is shut' not in text
    # the same for controllability: a record without gate_open is not a shut gate
    ctl = dict(measured_share=0.25, agreement_mean=0.3, control_mean=0.02, value_mean=0.12)
    text = '\n'.join(m.gof_lines([dict(type='mimo', controllability=ctl)]))
    assert 'evidence gate state unmeasured' in text and '0 while the gate is shut' not in text
    assert 'its running agreement +12%, if the gate is open (its state is not in this record)' in text


def test_liveness_and_pool_spread_print_against_the_section_10_line():
    """review dashboards M10 applied to the watcher: every liveness and pool_spread sub-block, and
    the top-level block, against DESIGN section 10's line; pool spread has no declared line."""
    m = _module()
    rec = dict(type='mimo', contact_control=dict(liveness={'trunk.0': 0.8, 'patch.0': 0.42},
                                                 pool_spread={'patch_pool': 0.091}),
               synergy=dict(liveness={'net.0': 1.0}),
               liveness=dict(layers={'tenure_heads.pose.0': 0.0, 'motor_controller.residual_blocks.0.0': 0.9},
                             pools={'motor_encoder.pool': 0.001}, line=-3, samples=24,
                             revived=['tenure_heads.pose.0']))
    text = '\n'.join(m.liveness_lines([rec]))
    assert 'contact_control: trunk.0 80 of 100, patch.0 42 of 100' in text
    assert 'whole model: tenure_heads.pose.0 0 of 100' in text
    assert 'LIVENESS ALERT: 2 layers under the line of 50 live units in 100' in text
    assert 'no line declared' in text and 'contact_control patch_pool 9.1%' in text
    assert 'whole model motor_encoder.pool 0.1%' in text
    assert 'revived by the repair' in text
    assert not bare_decimals(text), bare_decimals(text)
    # fix-readout-dashboards 2026-09-23: the line printed "units live = input above -3", a raw input
    # value of a hidden unit (the readout scanner's rule); it now says what the line means
    assert not unanchored_numbers(text), unanchored_numbers(text)
    assert ('a unit is live when its input leaves the flat low end of its activation function (GELU), where a '
            'change in the input moves the output by less than 1.2% of the change') in text
    assert 'input above -3' not in text
    # the input line the whole-model block says it used is the one described (GELU's slope at -2 is
    # 8.52%, rounded up so "less than" stays true)
    rec['liveness'] = dict(rec['liveness'], line=-2)
    text = '\n'.join(m.liveness_lines([rec]))
    assert 'less than 8.6% of the change' in text and 'input above -2' not in text
    rec['liveness'] = dict(rec['liveness'], line=0)
    assert 'the line the whole-model report names (field liveness.line)' in '\n'.join(m.liveness_lines([rec]))
    text = '\n'.join(m.liveness_lines([dict(type='mimo', synergy=dict(liveness={'net.0': 0.99}))]))
    assert 'LIVENESS ALERT' not in text and 'synergy: net.0 99 of 100' in text
    assert 'unmeasured' in m.liveness_lines([dict(type='mimo')])[0]


def test_a_line_builder_that_raises_never_stops_the_check(tmp_path, monkeypatch, capsys):
    """An informational line that raises prints LINE FAILED; the stop rules after it still run."""
    bad = dict(causal_chain=dict(alternatives_n='four', chosen_best_share=0.5))
    assert watch(tmp_path, monkeypatch, baseline=BASE, nan_steps=1, **bad)['cmd'] == 'snapshot_and_stop'
    assert 'non-finite' in json.loads((tmp_path / 'stop_reason.json').read_text())['reason']


def test_a_text_or_a_block_where_a_number_should_be_does_not_break_a_mean():
    m = _module()
    rr = [dict(x=dict(a=1.0)), dict(x=dict(a='n/a')), dict(x=dict(a=dict(b=2))), dict(x=None),
          dict(x=dict(a=True))]
    assert m.wmean(rr, 'x', 'a') == 1.0                                     # 1.0 and True
    assert m.wmean([dict(g=[1.0, 2.0, 3.0]), dict(g=[3.0])], 'g', '1') == 2.0  # list positions
    assert m.wmean([dict(g=[1.0])], 'g', '5') is None


def test_declared_criteria_read_the_new_fields_by_dot_path(tmp_path, monkeypatch, capsys):
    alerts = [dict(name='gaze_100ms', path='efference.gaze_deg_err.1', op='>', line=1.5, mode='alert', note='n'),
              dict(name='vision_before', path='causal_chain.pred_err_latch.vision', op='>', line=0.9,
                   mode='alert', note='n')]
    stop = dict(name='commit', path='machine.commit_gb', op='>', line=60.0, mode='stop', note='n')
    # the alerts are read and printed at the hourly sweep
    assert watch(tmp_path, monkeypatch, baseline=BASE, criteria=alerts,
                 machine=dict(commit_gb=61.0, commit_limit_gb=68.8), **gof_record()) is None
    out = capsys.readouterr().out
    assert "criterion 'gaze_100ms': efference.gaze_deg_err.1 reads 133% of its line (a reading over the line is a breach) -> BREACHED" in out
    assert "criterion 'vision_before': causal_chain.pred_err_latch.vision reads 89% of its line" in out and '-> holding' in out
    # a declared stop is read at the first 5-minute check (the relaunch review of Oct 5, 2026, finding F4), by dot path
    got = watch(tmp_path, monkeypatch, baseline=BASE, criteria=alerts + [stop],
                machine=dict(commit_gb=61.0, commit_limit_gb=68.8), **gof_record())
    assert got['cmd'] == 'snapshot_and_stop'
    why = json.loads((tmp_path / 'stop_reason.json').read_text())['reason']
    assert "declared criterion 'commit'" in why and "(5-minute check)" in why


def test_the_hourly_check_prints_the_new_blocks_and_the_machine(tmp_path, monkeypatch, capsys):
    assert watch(tmp_path, monkeypatch, baseline=BASE,
                 machine=dict(ram_available_gb=9.5, commit_gb=40.0, commit_limit_gb=68.8, rss_gb=24.0,
                              gpu_used_mib=7000, gpu_total_mib=8188, step_ms=800, model_ms=550,
                              env_ms=250), **gof_record()) is None
    out = capsys.readouterr().out
    assert 'causal chain: 30 commands predicted' in out and 'efference gaze head' in out
    # the check's second line gives every name the lines below use its plain meaning (the style guide "ONE WORD PER
    # THING": a model step, a command, and the senses by the registry's names with the explainer's own meanings)
    from mimo_bridge.explainer.schema import SENSE_DEFINITIONS
    names = [line for line in out.splitlines() if line.startswith("  names used below:")]
    assert names and len(set(names)) == 1          # once per hourly check, the same line each time
    assert "a model step = one model update" in names[0] and "a command = 9 numbers per body handle" in names[0]
    for sense, meaning in _module().SENSE_MEANS:
        assert SENSE_DEFINITIONS[sense].startswith(meaning) and "%s = %s" % (sense, meaning) in names[0], sense
    # and no gauge line uses the words these lines printed until 2026-10-03
    lines = [line for line in out.splitlines() if line.startswith("  ")]
    # ("touch" stays a word for a contact: "a real own-body touch onset"; it is refused where it named the sense)
    import re as _re
    for word in (r" orders?\b", r"moment", r"sight and touch", r"\btouch \d", r", touch\b", r"vestibular sense",
                 r"hearing", r"body position", r"balance sense", r"balance token"):
        assert not [line[:100] for line in lines
                    if _re.search(word, line.replace("the skin's sense of touch", ""))], word
    assert "machine (the trainer's records, hour means): RAM free for new work 9.5 GB" in out
    assert 'one model step 800 ms (550 ms model, 250 ms bodies)' in out
    assert 'machine (read now by this watcher): RAM available 64.0 GB' in out
    assert '28.8 GB of headroom' in out and 'COMMIT ALERT' not in out
    assert 'ARCHIVER ALERT: no readable status file' in out     # the test's empty Saved folder
    assert 'liveness (a unit is live when' in out and 'LIVENESS ALERT' in out   # gaze.0 31, patch.0 42
    assert 'LINE FAILED' not in out
    printed = [l for l in out.splitlines() if not l.startswith(('Monitoring', 'own baselines loaded'))]
    assert not bare_decimals('\n'.join(printed)), bare_decimals('\n'.join(printed))
    assert not unanchored_numbers('\n'.join(printed)), unanchored_numbers('\n'.join(printed))


def test_every_printed_line_on_a_real_segment_31_record_has_no_bare_decimal(tmp_path, monkeypatch):
    """DESIGN section 11 on the watcher's own output, read from the lineage's real stats: the last
    hour of segment 31's records (the run that stopped at step 344,400)."""
    stats = SEG31 / 'replay_training_stats.jsonl'
    if not stats.exists():
        pytest.fail("segment 31's stats are gone from %s; the scan needs one real record" % stats)
    with open(stats, 'rb') as fh:
        fh.seek(max(0, stats.stat().st_size - 3_000_000))
        tail = fh.read().decode('utf-8', 'replace').splitlines()[1:]
    recs = []
    for line in tail:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get('type') == 'mimo':
            recs.append(r)
    assert len(recs) >= 5, len(recs)
    m = _module()
    last = recs[-1]
    w = [r for r in recs if r.get('timestamp', 0) >= last['timestamp'] - 3600]
    m.RUN_ARGS = json.loads((SEG31 / 'args.json').read_text(encoding='utf-8'))
    healthy_resources(monkeypatch)
    lines = m.check_lines(3, w, last, dict(BASE, proprioception=0.004, audio=0.07), last['elapsed_s'] / 3600, 0.4)
    lines += m.hself_lines(w, last) + m.gof_lines(w) + m.liveness_lines(w)
    lines += m.storage_lines(saved=tmp_path, archive=tmp_path / 'archive')
    lines += m.machine_lines(w, m.live_machine())
    text = '\n'.join(lines)
    assert 'LINE FAILED' not in text, text
    assert not bare_decimals(text), bare_decimals(text)
    assert not unanchored_numbers(text), unanchored_numbers(text)


def test_commit_charge_inside_the_guards_reserve_alerts_but_does_not_stop():
    m = _module()
    live = dict(ram_available_gb=3.0, commit_gb=64.0, commit_limit_gb=68.8, gpu_used_mib=None,
                gpu_total_mib=None)
    text = '\n'.join(m.machine_lines([dict(type='mimo')], live))
    assert 'COMMIT ALERT: 4.8 GB between the commit charge and its limit' in text
    assert "machine (the trainer's records): unmeasured" in text and 'graphics card unmeasured' in text


def _manifests(archive, now, per_day_gb, days):
    (archive / 'caregiver_tapes').mkdir(parents=True, exist_ok=True)
    with open(archive / 'move_manifest.jsonl', 'w', encoding='utf-8') as fh:
        for d in range(days):
            fh.write(json.dumps(dict(event='moved', bytes=per_day_gb * 0.9e9,
                                     time=now - (d + 0.5) * 86400)) + '\n')
    with open(archive / 'caregiver_tapes' / 'tape_move_manifest.jsonl', 'w', encoding='utf-8') as fh:
        for d in range(days):
            fh.write(json.dumps(dict(event='moved', bytes=per_day_gb * 0.1e9,
                                     time=now - (d + 0.5) * 86400)) + '\n')
        fh.write('{"event": "moved", "bytes": 1')                          # a torn last line


def _alive(pid):
    return True, True


def test_the_archive_drives_runway_is_measured_from_the_manifests_and_alerts_under_five_days(tmp_path, monkeypatch):
    m = _module()
    now = 2_000_000_000.0
    saved, archive = tmp_path / 'saved', tmp_path / 'D' / 'archive'
    saved.mkdir()
    (saved / 'checkpoint_archive_status.json').write_text(json.dumps(
        dict(moved=2, free_gb=85.0, time=now - 300, tapes=dict(moved_files=4), pid=77, watch=True,
             dry_run=False)))
    _manifests(archive, now, per_day_gb=34.0, days=9)          # 34.3 GB a day, the survey's rate
    intake = m.archive_intake(archive, now)
    assert intake['day_gb'] == pytest.approx(34.0) and intake['span_days'] == 7.0
    assert intake['week_gb_per_day'] == pytest.approx(34.0)
    # a history shorter than the window: moves at 0.5, 1.5 and 2.5 days back are 2 moves after
    # the first over 2.5 days, not 3 moves over 2.5 days
    short = tmp_path / 'D' / 'short'
    _manifests(short, now, per_day_gb=34.0, days=3)
    assert m.archive_intake(short, now)['week_gb_per_day'] == pytest.approx(2 * 34.0 / 2.5)
    usage = collections.namedtuple('usage', 'total used free')
    monkeypatch.setattr(shutil, 'disk_usage', lambda _p: usage(1e12, 0.0, 510e9))
    text = '\n'.join(m.storage_lines(now, dict(caregiver_record_dir=str(tmp_path / 'rec')), saved, archive,
                                     process=_alive))
    assert 'runway 15.0 days' in text and 'STORAGE ALERT' not in text
    assert 'the caregiver-record drive' in text and 'stop line 8 GB' in text
    assert 'status was written 300 s ago by pid 77 (watching)' in text
    assert 'moved 2 checkpoints and 4 recording files' in text and 'ALERT' not in text
    monkeypatch.setattr(shutil, 'disk_usage', lambda _p: usage(1e12, 0.0, 150e9))
    text = '\n'.join(m.storage_lines(now, {}, saved, archive, process=_alive))
    assert 'STORAGE ALERT: the archive drive fills in 4.4 days' in text
    assert 'caregiver-record drive unmeasured' in text
    assert not bare_decimals(text), bare_decimals(text)


def test_a_silent_or_missing_archiver_alerts(tmp_path):
    m = _module()
    now = 2_000_000_000.0
    saved = tmp_path / 'saved'
    saved.mkdir()
    text = '\n'.join(m.storage_lines(now, {}, saved, tmp_path / 'no_archive', process=_alive))
    assert 'ARCHIVER ALERT: no readable status file' in text
    assert 'intake unmeasured' in text and 'runway unmeasured' in text
    status = saved / 'checkpoint_archive_status.json'
    status.write_text(json.dumps(dict(moved=0, free_gb=80.0, time=now - 1801, pid=77, watch=True)))
    text = '\n'.join(m.storage_lines(now, {}, saved, tmp_path / 'no_archive', process=_alive))
    assert "ARCHIVER ALERT: the archiver's status is 1801 s old, past 1800 s" in text
    status.write_text(json.dumps(dict(moved=0, free_gb=80.0, time=now - 1799, pid=77, watch=True)))
    assert 'ARCHIVER ALERT' not in '\n'.join(m.storage_lines(now, {}, saved, tmp_path / 'no_archive',
                                                            process=_alive))
    # a status from code older than the watch and pid fields: its age is judged alone, as before
    status.write_text(json.dumps(dict(moved=0, free_gb=80.0, time=now - 1801)))
    text = '\n'.join(m.storage_lines(now, {}, saved, tmp_path / 'no_archive', process=_alive))
    assert 'predates the watch and pid fields' in text and "status is 1801 s old" in text


def test_an_unreadable_archive_drive_is_an_alert_that_names_it(tmp_path, monkeypatch):
    """Review ops-storage finding 2: with the archive drive gone the watcher printed no alert and
    gave the wrong reason ('no move manifest readable')."""
    m = _module()
    now = 2_000_000_000.0
    saved = tmp_path / 'saved'
    saved.mkdir()
    (saved / 'checkpoint_archive_status.json').write_text(json.dumps(
        dict(moved=0, free_gb=80.0, time=now - 60, pid=77, watch=True)))
    real = shutil.disk_usage

    def gone(p):
        if str(p).upper().startswith('Q:'):
            raise FileNotFoundError(3, 'The system cannot find the path specified')
        return real(p)
    monkeypatch.setattr(shutil, 'disk_usage', gone)
    text = '\n'.join(m.storage_lines(now, {}, saved, Path('Q:/UEMA_recordings/H84_checkpoint_archive'),
                                     process=_alive))
    assert 'ARCHIVE DRIVE ALERT: the archive drive Q:\\ cannot be read' in text
    assert 'free space unreadable' in text and 'unmeasured GB' not in text
    assert 'no move manifest readable' not in text


@pytest.mark.parametrize('status,where', [
    (dict(error='[WinError 3] The system cannot find the path specified'), 'the checkpoint pass failed'),
    (dict(tapes=dict(moved_files=0, error='[Errno 28] No space left on device')), 'the recordings pass failed'),
])
def test_a_status_carrying_an_error_is_an_alert_that_quotes_it(tmp_path, status, where):
    """Review ops-storage finding 2: a pass that cannot write the archive still writes a fresh status."""
    m = _module()
    now = 2_000_000_000.0
    saved = tmp_path / 'saved'
    saved.mkdir()
    (saved / 'checkpoint_archive_status.json').write_text(json.dumps(
        dict(dict(moved=0, free_gb=80.0, time=now - 60, pid=77, watch=True), **status)))
    text = '\n'.join(m.storage_lines(now, {}, saved, tmp_path / 'arch', process=_alive))
    err = status.get('error') or status['tapes']['error']
    assert f'ARCHIVER ALERT: {where}: "{err}"' in text


def test_the_heartbeat_is_judged_only_on_a_live_watching_archivers_status(tmp_path):
    """Review ops-storage finding 2: a one-off or dry pass refreshes the status file and hid a dead
    watching archiver for up to 1,800 s; the age is now judged only when the status comes from a
    watching archiver whose process still runs."""
    m = _module()
    now = 2_000_000_000.0
    saved = tmp_path / 'saved'
    saved.mkdir()
    status = saved / 'checkpoint_archive_status.json'

    def lines(st, process, watchers):
        status.write_text(json.dumps(dict(dict(moved=0, free_gb=80.0), **st)))
        return '\n'.join(m.storage_lines(now, {}, saved, tmp_path / 'arch', process=process,
                                         watchers=watchers))
    # the watching archiver that wrote a fresh status has died
    text = lines(dict(time=now - 60, pid=77, watch=True), lambda p: (False, False), lambda: [])
    assert 'ARCHIVER ALERT: the watching archiver that wrote the status (pid 77) is not running' in text
    # a fresh one-off pass hides no dead watcher: with none running it is an alert
    text = lines(dict(time=now - 60, pid=88, watch=False), lambda p: (False, False), lambda: [])
    assert 'ARCHIVER ALERT: the newest status came from a one-off or dry pass and no watching archiver' in text
    # with a watching archiver running, a one-off status of any age is not judged
    text = lines(dict(time=now - 5000, pid=88, watch=False), lambda p: (False, False), lambda: [42])
    assert 'ALERT' not in text and 'pid 42 runs with --watch' in text
    # a dry run with --report is not the heartbeat either
    text = lines(dict(time=now - 60, pid=88, watch=True, dry_run=True), lambda p: (True, True), lambda: [])
    assert 'ARCHIVER ALERT: the newest status came from a one-off or dry pass' in text
    # the control: a live watching archiver's old status is still judged by age
    text = lines(dict(time=now - 1801, pid=77, watch=True), lambda p: (True, True), lambda: [77])
    assert "ARCHIVER ALERT: the archiver's status is 1801 s old" in text


def test_a_full_caregiver_record_drive_stops_the_run(tmp_path, monkeypatch):
    """The recorder and every save's episode companion write to caregiver_record_dir; that
    drive now has the same 8 GB stop as the stats and checkpoint drives."""
    healthy_resources(monkeypatch)
    monkeypatch.setenv('UEMA_WATCH_SAVED', str(tmp_path / '_saved'))
    monkeypatch.setenv('UEMA_WATCH_ARCHIVE', str(tmp_path / '_archive'))
    low = HEALTHY_DISK._replace(free=2e9)
    monkeypatch.setattr(shutil, 'disk_usage',
                        lambda p: low if str(p).upper().startswith('Q:') else HEALTHY_DISK)
    (tmp_path / 'args.json').write_text(json.dumps({'caregiver_record_dir': 'Q:\\recordings\\H99'}))
    (tmp_path / 'replay_training_stats.jsonl').write_text(json.dumps(row()) + '\n')
    monkeypatch.setattr(sys, 'argv', [str(WATCHER), str(tmp_path), '3'])
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    monkeypatch.setattr(time, 'time', lambda: 10000)
    runpy.run_path(str(WATCHER), run_name='__main__')
    assert json.loads((tmp_path / 'control.json').read_text())['cmd'] == 'snapshot_and_stop'
    reason = json.loads((tmp_path / 'stop_reason.json').read_text())['reason']
    assert 'Q:' in reason and '8 GB line' in reason


# ---- the machine lines inside a training pause (operator-approved fix, 2026-09-26) ----
def _held(tmp_path, monkeypatch, paused, alive, low='memory'):
    """One quick check with low memory (or disk), the supervisor paused or not, the recorded trainer alive or not."""
    import psutil
    healthy_resources(monkeypatch)
    saved = tmp_path / '_saved'
    saved.mkdir(exist_ok=True)
    monkeypatch.setenv('UEMA_WATCH_SAVED', str(saved))
    monkeypatch.setenv('UEMA_WATCH_ARCHIVE', str(tmp_path / '_archive'))
    (saved / 'supervisor_control.json').write_text(json.dumps({'cmd': 'pause' if paused else 'run'}))
    (saved / 'training_continuation.json').write_text(json.dumps({'active_run': str(tmp_path), 'trainer_pids': [4242]}))

    class FakeProcess:                       # never a real process named like a trainer: watchers scan for those
        def __init__(self, pid):
            if not alive:
                raise psutil.NoSuchProcess(pid)

        def cmdline(self):
            return ['python', '-m', 'mimo_bridge.train_mimo']

    monkeypatch.setattr(psutil, 'Process', FakeProcess)
    if low == 'memory':
        from mimo_bridge.tools import commit_guard
        monkeypatch.setattr(commit_guard, 'commit_reading', lambda: (82.0, 84.4, 1.0))  # 2.4 GB of commit room: under COMMIT_STOP_GB
    elif low == 'free_memory_only':
        # Sep 30 2026, 6:11 PM: 0.6 GB free with 11.6 GB of commit room left
        from mimo_bridge.tools import commit_guard
        monkeypatch.setattr(psutil, 'virtual_memory', lambda: collections.namedtuple('vm', 'available')(0.6e9))
        monkeypatch.setattr(commit_guard, 'commit_reading', lambda: (72.8, 84.4, 0.6))
    else:
        monkeypatch.setattr(shutil, 'disk_usage', lambda _p: HEALTHY_DISK._replace(free=1e9))
    for stale in ('control.json', 'stop_reason.json'):
        if (tmp_path / stale).exists():
            (tmp_path / stale).unlink()
    (tmp_path / 'args.json').write_text('{}')
    (tmp_path / 'replay_training_stats.jsonl').write_text(json.dumps(row()) + '\n')
    (tmp_path / 'watcher_baselines.json').write_text(json.dumps(BASE))
    monkeypatch.setattr(sys, 'argv', [str(WATCHER), str(tmp_path), '1'])
    monkeypatch.setattr(time, 'sleep', lambda _: None)
    monkeypatch.setattr(time, 'time', lambda: 10000)
    runpy.run_path(str(WATCHER), run_name='__main__')
    return (tmp_path / 'stop_reason.json').exists(), json.loads((saved / 'supervisor_control.json').read_text())


@pytest.mark.parametrize('low', ['memory', 'disk'])
def test_a_machine_line_in_a_pause_with_no_trainer_writes_no_stop_and_keeps_the_pause_owner(tmp_path, monkeypatch, capsys, low):
    stopped, control = _held(tmp_path, monkeypatch, paused=True, alive=False, low=low)
    assert not stopped and not (tmp_path / 'control.json').exists()
    assert control == {'cmd': 'pause'}           # the pause's owner is untouched, so it can still lift it
    assert 'NO TRAINER RUNNING' in capsys.readouterr().out


@pytest.mark.parametrize('paused, alive', [(True, True), (False, False), (False, True)])
def test_a_machine_line_still_stops_a_running_trainer_or_a_relaunch_without_a_pause(tmp_path, monkeypatch, paused, alive):
    stopped, _ = _held(tmp_path, monkeypatch, paused=paused, alive=alive)
    assert stopped
    assert json.loads((tmp_path / 'control.json').read_text())['cmd'] == 'snapshot_and_stop'


def test_a_dying_repair_layer_stops_when_no_repair_window_can_come():
    """--liveness-rule none (re-review 2026-09-26 section 4): no repair window starts, each repair layer's
    watching entry reads needed None, and a repair layer under the line in two complete reports is judged
    like every other layer. The memory trial's clean arm read the motor controller's first residual layer
    at 4 and then 3 in 100 live. A window still open (needed 800, not ready) keeps the skip, and so does an
    entry without a needed field (older reports)."""
    m = _module()
    layer = 'motor_controller.residual_blocks.0.0'

    def rep(step, share, watching):
        return dict(step=step, liveness=dict(mode='on', samples=112, window_steps=100, layers={layer: share},
                                             watching={layer: dict(watching)}))
    none = {'samples': 0, 'needed': None, 'ready': False}
    stop = m.liveness_breach([rep(825, .04, none), rep(925, .03, none)])
    assert stop and layer + ': 3 of 100 units responding' in stop
    assert m.liveness_breach([rep(825, .04, none), rep(925, .6, none)]) is None
    open_window = {'samples': 40, 'needed': 800, 'ready': False}
    assert m.liveness_breach([rep(825, .04, open_window), rep(925, .03, open_window)]) is None
    assert m.liveness_breach([rep(825, .04, {'ready': False}), rep(925, .03, {'ready': False})]) is None
    ended = {'samples': 800, 'needed': 800, 'ready': True}
    assert m.liveness_breach([rep(825, .04, ended), rep(925, .03, ended)])
    text = '\n'.join(m.liveness_lines([rep(925, .03, none)]))
    assert 'LIVENESS ALERT' in text and 'every layer is judged from its first two complete reports' in text


def test_the_efference_vision_layer_alerts_and_every_other_layer_still_stops(capsys):
    """2026-09-29, H84 segment 41: efference.lesson.vision_in stopped the run at 48 of 100 although its measured
    functional cost is none (removing its 52 inactive units: 0.006759 to 0.006758) and the coordinator's record says
    it does not block protected training. It prints an alert; another layer under the line still stops."""
    m = _module()

    def rep(step, vision, other):
        return dict(step=step, liveness=dict(mode='on', samples=112, window_steps=100,
                                             layers={'tenure_heads.pose.0': other}, watching={}),
                    efference=dict(liveness=dict(vision_in=vision)))
    assert m.liveness_breach([rep(16800, .48, .8), rep(16900, .47, .8)]) is None
    assert 'LIVENESS ALERT (not a stop): efference.lesson.vision_in: 47 of 100' in capsys.readouterr().out
    stop = m.liveness_breach([rep(16800, .48, .45), rep(16900, .47, .44)])
    assert stop and 'tenure_heads.pose.0: 44 of 100 units responding' in stop and 'vision_in' not in stop


def test_the_efference_body_layer_alerts_too(capsys):
    """2026-09-29, 6:49 PM: efference.lesson.body_mlp.0 at 50 of 100; its measured functional cost is none."""
    m = _module()

    def rep(step, body):
        return dict(step=step, liveness=dict(mode='on', samples=112, window_steps=100, layers={}, watching={}),
                    efference=dict(liveness=dict(body_mlp=dict(), **{'body_mlp.0': body})))
    assert m.liveness_breach([rep(28900, .49), rep(29000, .495)]) is None
    assert 'LIVENESS ALERT (not a stop): efference.lesson.body_mlp.0' in capsys.readouterr().out



def test_low_free_memory_with_commit_room_left_is_an_alert_not_a_stop(tmp_path, monkeypatch, capsys):
    """Operator, Sep 30 2026: free memory under 1.5 GB during hourly saves stopped training three times with 7.1 to
    11.6 GB of commit room left; only commit room under COMMIT_STOP_GB (4 GB) stops a running trainer now."""
    stopped, control = _held(tmp_path, monkeypatch, paused=False, alive=True, low='free_memory_only')
    assert not stopped and control == {'cmd': 'run'}
    assert 'MEMORY ALERT (not a stop)' in capsys.readouterr().out


def test_commit_room_under_the_line_stops_a_running_trainer(tmp_path, monkeypatch):
    stopped, control = _held(tmp_path, monkeypatch, paused=False, alive=True, low='memory')
    assert stopped and 'commit room' in json.loads((tmp_path / 'stop_reason.json').read_text())['reason']
