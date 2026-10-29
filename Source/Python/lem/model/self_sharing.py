# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Many-Minds boundary: share completed percepts, count physical functioning once.

These contracts can be used by a later split without changing the current brain.
They do not themselves create an additional mind or establish consciousness.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CompletedSelf:
    body_id: str
    mind_id: str
    time: float
    percept: torch.Tensor
    physical_sites: tuple
    functioning: torch.Tensor


def read_completed(reports,now):
    """Self-reading uses an already completed state, never this step's answer."""
    result=[]
    for report in reports:
        if report.time>=now:raise ValueError('inter-mind reading requires a completed earlier perception')
        if not torch.isfinite(report.percept).all():raise ValueError('invalid inter-mind self perception')
        result.append(report.percept.detach().clone())
    return result


def unique_physical_function(reports):
    """Repeated copies of the same physical need cannot multiply its motivation."""
    result={}
    for report in reports:
        if len(report.physical_sites)!=len(report.functioning):raise ValueError('site/function mismatch')
        if not torch.isfinite(report.functioning).all():raise ValueError('invalid physical functioning')
        for site,value in zip(report.physical_sites,report.functioning):
            key=(report.body_id,site,report.time)
            if key in result and not torch.equal(result[key],value):
                raise ValueError('minds disagree about the same physical observation; resolve provenance')
            result.setdefault(key,value)
    return result
