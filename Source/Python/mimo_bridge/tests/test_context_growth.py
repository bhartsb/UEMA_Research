# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import torch
import pytest
from mimo_bridge.tools.grow_context import grow_table


def test_old_positions_and_moments_keep_their_time_ago():
    old=torch.arange(12.).reshape(4,3)
    new=grow_table(old,9)
    for age in range(4): torch.testing.assert_close(new[-1-age],old[-1-age])
    torch.testing.assert_close(new[:5],old[0].expand(5,3))
    moments=grow_table(old,9,moments=True)
    assert moments[:5].count_nonzero() == 0
    torch.testing.assert_close(moments[-4:],old)


def test_growth_refuses_to_discard_old_positions():
    with pytest.raises(ValueError): grow_table(torch.ones(4,3),3)
