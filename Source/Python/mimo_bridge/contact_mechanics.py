# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""SI mechanics for explicit hypothetical scenarios, not human pain predictions."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
import math

G = 9.80665


def positive(**values):
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')


def pressure_pa(force_n, area_m2):
    positive(area_m2=area_m2)
    if not math.isfinite(force_n) or force_n < 0:
        raise ValueError('force_n must be finite and nonnegative')
    return force_n / area_m2


def stopped_impact(mass_kg, speed_m_s, stopping_m, area_m2, gravity_along_motion=0.):
    """Constant deceleration, zero rebound, average force over stopping displacement.

    The same force is the time average ONLY for this constant-force assumption.
    Gravity along motion is positive for a downward fall. Effective mass is an
    explicit scenario input, not inferred from a person's total mass.
    """
    positive(mass_kg=mass_kg, speed_m_s=speed_m_s, stopping_m=stopping_m, area_m2=area_m2)
    if not math.isfinite(gravity_along_motion):
        raise ValueError('gravity must be finite')
    energy = .5 * mass_kg * speed_m_s ** 2
    force = energy / stopping_m + mass_kg * gravity_along_motion
    positive(contact_force_n=force)
    duration = 2 * stopping_m / speed_m_s
    # Momentum and work are computed separately to expose missing gravity/units.
    impulse = mass_kg * speed_m_s + mass_kg * gravity_along_motion * duration
    work = force * stopping_m - mass_kg * gravity_along_motion * stopping_m
    return dict(force_n=force, pressure_pa=pressure_pa(force, area_m2), energy_j=energy,
                duration_s=duration, contact_impulse_ns=impulse,
                impulse_residual_ns=force * duration - impulse,
                work_residual_j=work - energy, peak_force_n=None,
                model='constant deceleration, no rebound; peak not estimated')


def falling_impact(mass_kg, height_m, stopping_m, area_m2):
    positive(height_m=height_m)
    return stopped_impact(mass_kg, math.sqrt(2 * G * height_m), stopping_m, area_m2, G)


def surface_point_velocity(linear, angular, offset):
    """Velocity at contact = velocity at reference point + omega cross offset."""
    import numpy as np
    return np.asarray(linear) + np.cross(angular, offset)


def conservative_spread_linear(touch, contact_id, body_id, force):
    """Distribute a vector by geometric weights, independently of its direction.

    MIMo's original spread normalizes only component zero and drops a vector
    whose x component is zero. Conservation must hold for every orientation.
    """
    import numpy as np
    scale=float(getattr(touch,'force_spread_scales',touch.sensor_scales)[body_id])
    pos=touch.get_contact_position_relative(contact_id=contact_id,body_id=body_id)
    ids,dist=touch.get_sensors_within_distance(pos,body_id,3*scale)
    ids=np.asarray(ids,dtype=int);weights=np.maximum(1-np.asarray(dist)/(2*scale),0)
    if len(ids)==0 or weights.sum() <= 0:
        points=np.asarray(touch.sensor_positions[body_id])
        if len(points)==0: raise ValueError('contacting body has no skin samples')
        ids=np.array([np.linalg.norm(points-pos,axis=1).argmin()]);weights=np.ones(1)
    weights=weights/weights.sum()
    touch.sensor_outputs[body_id][ids] += weights[:,None]*np.asarray(force).reshape(1,-1)


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    mass_kg: tuple[float, float]
    travel: tuple[float, float]
    stopping_m: tuple[float, float]
    area_cm2: tuple[float, float]
    assumptions: str

    def calculate(self):
        rows = []
        for m, travel, stop, area in product(self.mass_kg, self.travel, self.stopping_m, self.area_cm2):
            area_m2 = area * 1e-4
            if self.kind == 'support':
                r = dict(force_n=m * G, pressure_pa=pressure_pa(m * G, area_m2))
            elif self.kind == 'fall':
                r = falling_impact(m, travel, stop, area_m2)
            elif self.kind == 'horizontal':
                r = stopped_impact(m, travel, stop, area_m2)
            else:
                raise ValueError(self.kind)
            rows.append(r)
        keys = ('force_n', 'pressure_pa', 'energy_j', 'duration_s', 'contact_impulse_ns')
        return dict(**asdict(self),
                    range={k: [min(r[k] for r in rows), max(r[k] for r in rows)]
                           for k in keys if k in rows[0]},
                    maximum_work_residual_j=max(abs(r.get('work_residual_j', 0)) for r in rows),
                    maximum_impulse_residual_ns=max(abs(r.get('impulse_residual_ns', 0)) for r in rows),
                    pain_rating=None, injury_probability=None,
                    interpretation='Illustrative mechanical envelope, not a probability interval or toddler pain threshold')


SCENARIOS = [
    Scenario('Standing on both feet', 'support', (10, 15), (0, 0), (0, 0), (60, 100),
             'Entire illustrative body weight shared over both soles; uniform average pressure.'),
    Scenario('Resting on a broad padded support', 'support', (10, 15), (0, 0), (0, 0), (300, 600),
             'Entire body supported; area is assumed effective loaded area, not pad dimensions.'),
    Scenario('Short fall onto yielding support', 'fall', (10, 15), (.1, .3), (.03, .08), (100, 300),
             'Whole effective mass stopped vertically; includes pad and body compliance; no rebound.'),
    Scenario('Short fall onto firm support', 'fall', (10, 15), (.1, .3), (.005, .02), (20, 100),
             'Whole effective mass and concentrated support are illustrative bounds, not a specific injury reconstruction.'),
    Scenario('Hand or knee catches part of a fall', 'fall', (2, 6), (.1, .3), (.005, .03), (2, 15),
             'Only the specified effective mass is stopped at this contact; load sharing and joint flexion vary.'),
    Scenario('Light soft object bumps the body', 'horizontal', (.02, .08), (.5, 2), (.01, .04), (5, 20),
             'Moving object effective mass; yielding target and object together set stopping distance.'),
    Scenario('Rigid toy bumps the body', 'horizontal', (.1, .3), (.5, 2), (.002, .01), (.5, 4),
             'No powered follow-through; assumed target deformation and footprint; zero rebound.'),
    Scenario('Load against a broad edge', 'support', (.5, 2), (0, 0), (0, 0), (1, 4),
             'Equivalent supported load on a rounded edge; area must describe the deformed footprint.'),
    Scenario('Same load against a narrow edge', 'support', (.5, 2), (0, 0), (0, 0), (.02, .2),
             'Finite narrow footprint assumed. Not a cutting model; skin yielding can enlarge this area.'),
]
