# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Convert requested observation times using the simulator's actual step size."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ProbeClock:
    physics_step_seconds: float
    action_repeat: int
    motor_hold: int

    def __post_init__(self):
        if not math.isfinite(self.physics_step_seconds) or self.physics_step_seconds <= 0:
            raise ValueError('physics step must be finite and positive')
        if self.action_repeat < 1 or self.motor_hold < 1:
            raise ValueError('repeat and hold must be positive')

    @property
    def model_step_seconds(self):
        return self.physics_step_seconds * self.action_repeat

    @property
    def held_order_seconds(self):
        return self.model_step_seconds * self.motor_hold

    def steps_for(self, seconds):
        count = round(seconds / self.physics_step_seconds)
        if count < 1 or not math.isclose(count * self.physics_step_seconds, seconds,
                                         rel_tol=0., abs_tol=1e-9):
            raise ValueError(f'{seconds} seconds is not an integral simulator interval')
        return count

    def verify_elapsed(self, start, end, requested):
        elapsed = end - start
        if not math.isclose(elapsed, requested, rel_tol=0., abs_tol=1e-7):
            raise AssertionError(f'outcome requested at {requested}s, observed at {elapsed}s')
        return elapsed
