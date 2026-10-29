# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The machine block (DESIGN.md section 3, always on): memory, commit charge, card memory by
all processes, page faults and step time, every stats window."""
from mimo_bridge.machine_stats import MachineSampler, nvidia_smi_memory

FIELDS = {"ram_available_gb", "commit_gb", "commit_limit_gb", "rss_gb", "gpu_used_mib",
          "gpu_total_mib", "step_ms", "model_ms", "env_ms"}


def test_every_design_field_is_present_and_step_time_is_the_sum():
    calls = []
    s = MachineSampler(gpu_every=10, gpu_reader=lambda: calls.append(1) or (3000.0, 8188.0))
    rec = s.sample(model_ms=600.0, env_ms=200.0)
    assert FIELDS <= set(rec)
    assert rec["step_ms"] == 800.0 and rec["model_ms"] == 600.0 and rec["env_ms"] == 200.0
    assert rec["gpu_used_mib"] == 3000.0 and rec["gpu_total_mib"] == 8188.0
    assert rec["gpu_reading_age_windows"] == 0
    assert rec["rss_gb"] is not None and rec["rss_gb"] > 0
    assert rec["ram_available_gb"] is not None and rec["ram_available_gb"] > 0


def test_the_card_is_read_once_every_tenth_window_and_the_age_says_so():
    calls = []
    s = MachineSampler(gpu_every=10, gpu_reader=lambda: calls.append(1) or (1.0, 2.0))
    ages = [s.sample(1.0, 1.0)["gpu_reading_age_windows"] for _ in range(21)]
    assert len(calls) == 3                      # windows 0, 10, 20
    assert ages[:3] == [0, 1, 2] and ages[10] == 0 and ages[19] == 9 and ages[20] == 0


def test_an_unreadable_card_reports_none_not_zero():
    s = MachineSampler(gpu_reader=lambda: None)
    rec = s.sample(None, None)
    assert rec["gpu_used_mib"] is None and rec["gpu_reading_age_windows"] is None
    assert rec["step_ms"] is None


def test_page_faults_are_a_rate_between_windows():
    t = [100.0]
    s = MachineSampler(gpu_reader=lambda: None, clock=lambda: t[0])
    first = s.sample(1.0, 1.0)
    t[0] = 110.0
    second = s.sample(1.0, 1.0)
    assert first["page_faults_per_s"] is None
    if s._proc is not None and hasattr(s._proc.memory_info(), "num_page_faults"):
        assert second["page_faults_per_s"] is not None and second["page_faults_per_s"] >= 0


def test_nvidia_smi_reader_returns_a_pair_or_none():
    r = nvidia_smi_memory(timeout_s=10.0)
    assert r is None or (len(r) == 2 and r[1] > 0 and 0 <= r[0] <= r[1])


def test_the_causal_chains_time_is_part_of_the_step_and_named():
    s = MachineSampler(gpu_reader=lambda: None)
    on = s.sample(600.0, 200.0, causal_ms=40.0)
    assert on["step_ms"] == 840.0 and on["causal_chain_ms"] == 40.0
    off = s.sample(600.0, 200.0)
    assert off["step_ms"] == 800.0 and "causal_chain_ms" not in off


def test_the_efference_lessons_time_is_named_and_inside_the_models():
    """Review of b1, MINOR 6: the efference lesson had no timing of its own. It is reported beside
    model_ms and never added to step_ms (it is inside model_ms)."""
    s = MachineSampler(gpu_reader=lambda: None)
    on = s.sample(600.0, 200.0, efference_ms=12.345)
    assert on["efference_ms"] == 12.35 and on["step_ms"] == 800.0
    off = s.sample(600.0, 200.0)
    assert "efference_ms" not in off
