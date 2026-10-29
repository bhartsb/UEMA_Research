# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Reproduce physical scenario envelopes and their independent unit checks."""
import argparse
import json
from pathlib import Path
from mimo_bridge.contact_mechanics import SCENARIOS, G, pressure_pa, falling_impact


def run(out):
    rows = [s.calculate() for s in SCENARIOS]
    checks = dict(one_newton_one_cm2_pa=pressure_pa(1., 1e-4),
                  child_study_21_94_N_one_cm2_kpa=pressure_pa(21.94, 1e-4)/1000,
                  standing_12kg_80cm2_kpa=pressure_pa(12*G, 80e-4)/1000,
                  fall_12kg_20cm_stop4cm=falling_impact(12, .2, .04, .01))
    assert checks['one_newton_one_cm2_pa'] == 10000
    assert abs(checks['fall_12kg_20cm_stop4cm']['force_n'] - 12*G*6) < 1e-10
    assert all(r['maximum_impulse_residual_ns'] < 1e-10 and r['maximum_work_residual_j'] < 1e-10 for r in rows)
    result = dict(scenarios=rows, checks=checks, gravity_m_s2=G,
                  child_reference=dict(pressure_pa=219400, sd_pa=75100, sample_size_table=16,
                    ages_years=[2,5], tip_cm2=1, source='https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.963765/full',
                    limitation='First-discomfort mean in one subgroup; not a sharp-contact or injury law; summary/table counts differ'))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out/'contact_scenarios.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Assumed situations and calculated loads', '',
             'These are mechanical envelopes, not predictions that a particular child will feel pain. All dimensions are explicit assumptions. Forces below are constant-deceleration averages, not impact peaks.', '',
             '| Situation | Force (N) | Mean contact pressure (kPa) |', '|---|---:|---:|']
    for r in rows:
        f,p = r['range']['force_n'], r['range']['pressure_pa']
        lines.append(f"| {r['name']} | {f[0]:.2f}–{f[1]:.2f} | {p[0]/1000:.2f}–{p[1]/1000:.2f} |")
    lines += ['', '## Assumptions behind those ranges', '',
              'These are ranges of assumed inputs, not measured probabilities. A 100 N force is approximately the weight of a 10.2 kg mass under Earth gravity. Impacts also depend on how far the body and surface yield.', '',
              '| Situation | Effective mass (kg) | Fall height / impact speed | Stopping distance (cm) | Loaded area (cm²) |',
              '|---|---:|---|---:|---:|']
    for r in rows:
        span=lambda xs: f'{xs[0]:g}–{xs[1]:g}'
        travel=(span([v*100 for v in r['travel']])+' cm fall' if r['kind']=='fall' else
                span(r['travel'])+' m/s' if r['kind']=='horizontal' else 'static load')
        stopping=span([v*100 for v in r['stopping_m']]) if r['kind']!='support' else '—'
        lines.append(f"| {r['name']} | {span(r['mass_kg'])} | {travel} | {stopping} | {span(r['area_cm2'])} |")
    lines += ['', 'Full assumptions, stopping distances, masses, areas, durations and conservation checks are in [the calculation receipt](contact_scenarios.json).', '',
              'Example: a 12 kg effective mass falling 20 cm and stopping over 4 cm gives 706.08 N average contact force. Omitting gravity during stopping gives the wrong contact force. Increasing stopping distance lowers the force; decreasing contact area raises pressure without changing that force.', '',
              'A study of first discomfort in ages 2–5 reported 21.94 N with a 1 cm² tip: 219.4 kPa, not 21.94 kPa. The reported SD converts to 75.1 kPa. These are one group’s results under one procedure, not a universal pain boundary. [Primary study](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2022.963765/full).', '',
              'Work and impulse checks use the same explicitly constant deceleration model. For real force pulses, displacement-average force and time-average force need not be equal. [Impulse definition](https://openstax.org/books/university-physics-volume-1/pages/9-2-impulse-and-collisions).', '',
              'The narrow-edge values assume a finite deformed area. MuJoCo point contacts do not measure biological contact area; local tissue pressure and cutting remain unresolved without an appropriate skin/contact model. [MuJoCo contact model](https://mujoco.readthedocs.io/en/stable/computation/).']
    (out/'contact_scenarios.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    a=ap.parse_args(); run(a.out)
